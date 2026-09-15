#!/usr/bin/env python3
"""Weather2K Spatial-adapter STDK + location-specific 5-to-5 QConvLSTM.

Each seed samples 600 stations, trains only on train500, and evaluates on a
strict held-out100 whose observations never enter either training stage.
The front model keeps the Weather2K Spatial-adapter architecture, while the
probabilistic flow follows the author-data reconstruction: fit q50/q05/q95
STDK with pinball loss, form quantile-specific local grids and target series,
then fit a separate released three-block QConvLSTM at every target location,
as formulated for s_0 in the paper, with paper Eq. (7). The Weather2K split is
an explicit dataset adaptation. No model checkpoint is written.
"""
from __future__ import annotations

import argparse, csv, importlib.util, json, math, random, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT / "STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py"
SPATIAL_STDK = ROOT / "spatial-adapter/examples/baselines/stdk/st_interp.py"
DATA = ROOT / "Josh's Weather2K/Weather2K/weather2k.npy"
OUTPUT = HERE
BEST_PARAMS = HERE / "2K_best_stdk_qconvlstm_params_500to100.json"
SPATIAL_ADAPTER_ROOT = ROOT / "spatial-adapter"
if str(SPATIAL_ADAPTER_ROOT) in sys.path:
    sys.path.remove(str(SPATIAL_ADAPTER_ROOT))
sys.path.insert(0, str(SPATIAL_ADAPTER_ROOT))
from examples.baselines.stdk.utils.ema import ModelEMA
VARIABLES = ("air_pressure", "air_temperature", "relative_humidity", "wind_speed",
             "wind_direction", "precipitation", "solar_radiation",
             "dew_point_temperature", "cloud_cover", "visibility")

def load_base():
    spec = importlib.util.spec_from_file_location("stdk_qconv_base", BASE)
    if spec is None or spec.loader is None: raise ImportError(BASE)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod); return mod

def load_spatial_stdk():
    spec = importlib.util.spec_from_file_location("spatial_adapter_stdk", SPATIAL_STDK)
    if spec is None or spec.loader is None: raise ImportError(SPATIAL_STDK)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod); return mod

@dataclass
class Config:
    seed: int; n_sample: int = 600; n_train: int = 500; n_heldout: int = 100
    n_last: int = 1000; train_times: int = 700; val_times: int = 150
    test_times: int = 150; lookback: int = 5; horizon: int = 5
    grid_size: int = 8; radius: float = .2; stdk_epochs: int = 350
    qconv_epochs: int = 25; stdk_batch: int = 512; qconv_batch: int = 64
    qconv_lr: float = 1e-3; qconv_weight_decay: float = 0.0
    qconv_patience: int = 5; conv_filters: int = 64
    variable: str = "air_temperature"; forecast_mode: str = "block5to5"
    prediction_mode: str = "direct"
    scenario: str = "all_three"
    stdk_backend: str = "spatial_adapter"
    normalization_source: str = "obs100_train700"
    stdk_q50_loss: str = "pinball"
    qconv_training_target: str = "quantile_specific_fitted_stdk"
    qconv_model_scope: str = "location_specific"
    q50_checkpoint_selection: str = "pinball"
    stdk_validation_aggregation: str = "batch_mean"
    stdk_use_ema: bool = True
    validation_only: bool = False; q50_only: bool = False; smoke: bool = False

def model_name(cfg):
    return (
        "Spatial-adapter STDK+location-specific-residual-QConvLSTM"
        if cfg.prediction_mode == "residual" else
        "STDK+location-specific-QConvLSTM"
    )

def output_prefix(cfg):
    return (
        "location_specific_residual_qconvlstm"
        if cfg.prediction_mode == "residual" else
        "location_specific_qconvlstm"
    )

def output_stem(cfg):
    stem=f"{output_prefix(cfg)}_{cfg.forecast_mode}"
    if cfg.scenario != "all_three": stem+=f"_{cfg.scenario}"
    return f"{stem}_seed{cfg.seed}"

def spatial_stdk_config(cfg):
    """The exact STDK configuration used by 2K_STDK_500train_100test.py."""
    return {
        "p_covariates": 0, "regression_type": "mean",
        "epochs": cfg.stdk_epochs, "lr": 1e-3, "weight_decay": 1e-4,
        "batch_size": cfg.stdk_batch, "patience": 30,
        "use_ema": cfg.stdk_use_ema,
        "k_spatial_centers": [25, 81, 121],
        "k_temporal_centers": [10, 15, 45],
        "hidden_dims": [256, 256, 128], "dropout": 0.1,
        "layernorm": True, "spatial_learnable": False,
        "spatial_init_method": "uniform", "spatial_basis_function": "wendland",
        "gradient_damping": False, "damping_threshold": 0.0,
        "damping_strength": 1.0, "temporal_bandwidth_factor": 2.5,
        "use_delta_reparameterization": False,
    }

def flatten_stdk_points(coords, times, targets, total_times):
    """Time-major tensors matching the standalone Spatial-adapter STDK baseline."""
    coords=np.asarray(coords,dtype=np.float32); times=np.asarray(times,dtype=np.int64)
    coords_flat=np.tile(coords,(len(times),1)).astype(np.float32)
    denom=max(1,total_times-1)
    time_flat=np.repeat((times.astype(np.float32)/denom)[:,None],len(coords),axis=0)
    if targets is None: values=np.zeros((len(coords_flat),),dtype=np.float32)
    else: values=np.asarray(targets,dtype=np.float32)[:,times].T.reshape(-1)
    return coords_flat,time_flat.astype(np.float32),values

def fit_spatial_stdk(base, spatial, coords, scaled_values, train_times, val_times,
                     quantile, cfg, device, qlambda, median_model=None):
    """Fit quantile STDK using the reconstruction's pinball-loss workflow."""
    config=spatial_stdk_config(cfg)
    tr_coords,tr_time,tr_y=flatten_stdk_points(
        coords,train_times,scaled_values,cfg.n_last
    )
    va_coords,va_time,va_y=flatten_stdk_points(
        coords,val_times,scaled_values,cfg.n_last
    )
    train_ds=torch.utils.data.TensorDataset(
        torch.from_numpy(tr_coords),torch.from_numpy(tr_time),torch.from_numpy(tr_y[:,None])
    )
    seed_offset=0 if quantile==.5 else int(quantile*100)
    generator=torch.Generator().manual_seed(cfg.seed+1000+seed_offset)
    loader=DataLoader(train_ds,batch_size=cfg.stdk_batch,shuffle=True,generator=generator)
    torch.manual_seed(cfg.seed+seed_offset); np.random.seed(cfg.seed+seed_offset)
    model=spatial.create_model(config,train_coords=tr_coords).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
    use_ema=bool(config.get("use_ema",True))
    ema_decay=1.0-1.0/(10.0*len(loader))
    ema=ModelEMA(model,decay=ema_decay) if use_ema else None
    vc=torch.from_numpy(va_coords); vt=torch.from_numpy(va_time)
    vy=torch.from_numpy(va_y[:,None])
    if median_model is not None:
        median_model.eval()
        for parameter in median_model.parameters(): parameter.requires_grad_(False)
    best=float("inf"); state=None; wait=0; best_epoch=0
    for epoch in range(1,cfg.stdk_epochs+1):
        model.train()
        for batch_coords,batch_time,batch_y in loader:
            batch_coords=batch_coords.to(device); batch_time=batch_time.to(device); batch_y=batch_y.to(device)
            empty=batch_coords.new_empty((len(batch_coords),0)); optimizer.zero_grad(set_to_none=True)
            raw=model(empty,batch_coords,batch_time)
            if quantile==.5:
                prediction=raw; loss=base.pinball(prediction,batch_y,quantile)
            else:
                with torch.no_grad(): center=median_model(empty,batch_coords,batch_time)
                prediction=base.constrained_quantile(raw,center,quantile,qlambda)
                loss=base.pinball(prediction,batch_y,quantile)
            loss.backward(); optimizer.step()
            if ema is not None: ema.update(model)
        model.eval()
        if ema is not None: ema.apply_shadow()
        val_total=0.0; val_batches=0
        with torch.no_grad():
            for start in range(0,len(vc),cfg.stdk_batch):
                c=vc[start:start+cfg.stdk_batch].to(device)
                t=vt[start:start+cfg.stdk_batch].to(device)
                y=vy[start:start+cfg.stdk_batch].to(device)
                empty=c.new_empty((len(c),0)); raw=model(empty,c,t)
                if quantile==.5:
                    prediction=raw
                    batch_loss=base.pinball(prediction,y,quantile)
                else:
                    center=median_model(empty,c,t)
                    prediction=base.constrained_quantile(raw,center,quantile,qlambda)
                    batch_loss=base.pinball(prediction,y,quantile)
                val_total+=float(batch_loss); val_batches+=1
        val_loss=val_total/max(1,val_batches)
        if epoch==1 or epoch%10==0:
            print(f"Spatial-adapter STDK q={quantile:.2f} epoch={epoch:03d} val_pinball={val_loss:.6f}")
        if val_loss < best-1e-12:
            best=val_loss; wait=0; best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        if ema is not None: ema.restore()
        if wait>=config["patience"]: break
    if state is not None: model.load_state_dict(state)
    if median_model is not None:
        for parameter in median_model.parameters(): parameter.requires_grad_(True)
    return model.eval(),{"quantile":quantile,"best_validation_loss":best,
                         "best_epoch":best_epoch,"validation_loss_name":"pinball",
                         "validation_aggregation":"batch_mean","use_ema":use_ema,
                         "ema_decay":ema_decay if use_ema else None}

def predict_spatial_stdk(model, coords, times, cfg, device):
    coords_flat,time_flat,_=flatten_stdk_points(coords,times,None,cfg.n_last)
    predictions=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(coords_flat),cfg.stdk_batch):
            c=torch.from_numpy(coords_flat[start:start+cfg.stdk_batch]).to(device)
            t=torch.from_numpy(time_flat[start:start+cfg.stdk_batch]).to(device)
            predictions.append(model(c.new_empty((len(c),0)),c,t).cpu().numpy().reshape(-1))
    return np.concatenate(predictions).reshape(len(times),len(coords)).astype(np.float32)

class SharedWindows(Dataset):
    """Window dataset; formal QConvLSTM fits pass exactly one target station."""
    def __init__(self, frames, median, targets, lookback, horizon, qlambda):
        assert frames.shape == median.shape and frames.shape[:2] == targets.shape
        self.x = torch.from_numpy(frames[:, :, None]); self.mx = torch.from_numpy(median[:, :, None])
        self.y = torch.from_numpy(targets); self.lb = lookback; self.h = horizon
        self.nw = targets.shape[1] - lookback - horizon + 1; self.qlambda = qlambda
        if self.nw < 1: raise ValueError("train_times must exceed lookback+horizon")
    def __len__(self): return self.y.shape[0] * self.nw
    def __getitem__(self, i):
        station, start = divmod(i, self.nw); stop = start + self.lb
        return self.x[station,start:stop], self.y[station,stop:stop+self.h], self.mx[station,start:stop]


class SharedValidation(Dataset):
    """One Val150 forecast origin per supervised training station."""
    def __init__(self, frames, median, targets, qlambda):
        assert frames.shape == median.shape
        assert frames.shape[0] == targets.shape[0]
        self.x = torch.from_numpy(frames[:, :, None])
        self.mx = torch.from_numpy(median[:, :, None])
        self.y = torch.from_numpy(targets)
        self.qlambda = qlambda

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, i): return self.x[i], self.y[i], self.mx[i]


def block_history_windows(base, models, centers, target_start, target_length,
                          cfg, device):
    """Precompute STDK histories and return station-major block windows."""
    starts=list(range(target_start,target_start+target_length,cfg.horizon))
    histories=np.concatenate([
        np.arange(start-cfg.lookback,start,dtype=np.int64) for start in starts
    ])
    frames=frames_for(base,models,centers,histories,cfg,device)
    n_centers=len(centers); n_blocks=len(starts)
    windows={
        q: frames[q].reshape(n_centers,n_blocks,cfg.lookback,cfg.grid_size,cfg.grid_size)
          .reshape(n_centers*n_blocks,cfg.lookback,cfg.grid_size,cfg.grid_size)
        for q in frames
    }
    return windows,starts


def block_targets(targets, horizon):
    """Split station-major targets into station-major contiguous blocks."""
    if targets.shape[1] % horizon:
        raise ValueError("target length must be divisible by block horizon")
    return targets.reshape(targets.shape[0],targets.shape[1]//horizon,horizon).reshape(-1,horizon)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_weather(path, variable):
    raw = np.load(path, mmap_mode="r")
    if raw.ndim != 3 or raw.shape[1] < 13: raise ValueError(f"Weather2K shape={raw.shape}")
    coords = np.column_stack((raw[:,1,0], raw[:,0,0])).astype(np.float32)
    values = np.asarray(raw[:,3+VARIABLES.index(variable),:], dtype=np.float32)
    if not np.isfinite(values).all(): raise ValueError("non-finite target")
    return coords, values

def norm_coords(x):
    lo=x.min(0); span=x.max(0)-lo; span[span==0]=1
    return ((x-lo)/span).astype(np.float32)

def frames_for(base, models, centers, times, cfg, device):
    quantiles=tuple(q for q in (.05,.5,.95) if q in models)
    if .5 not in quantiles:
        raise ValueError("The median STDK model is required to build local grids")
    out={q:[] for q in quantiles}
    for i, center in enumerate(centers, 1):
        grid=base.regular_neighbourhood(center,cfg.grid_size,cfg.radius,"shift")
        def raw_prediction(model):
            return predict_spatial_stdk(
                model,grid,np.asarray(times),cfg,device
            ).reshape(len(times),cfg.grid_size,cfg.grid_size)
        median_grid=raw_prediction(models[.5])
        for q in out:
            if q==.5: grid_q=median_grid
            else:
                raw=raw_prediction(models[q]); deviation=models["lambda"]*abs(q-.5)/(1.+np.exp(-raw))
                grid_q=median_grid-deviation if q<.5 else median_grid+deviation
            out[q].append(grid_q.astype(np.float32))
        if i==1 or i%50==0 or i==len(centers): print(f"local grids {i}/{len(centers)}")
    return {q:np.stack(v).astype(np.float32) for q,v in out.items()}

def predict_stdk_quantiles(base, models, centers, times, cfg, device):
    """Direct Spatial-adapter STDK quantiles at station coordinates.

    This is used only as the causal warm-up for spatial evaluation times that
    do not have five earlier Weather2K frames.  It never reads held-out values.
    """
    times=np.asarray(times,dtype=np.int64)
    median=predict_spatial_stdk(models[.5],centers,times,cfg,device)
    quantiles=[]
    for q in (.05,.5,.95):
        if q==.5:
            prediction=median
        else:
            raw=predict_spatial_stdk(models[q],centers,times,cfg,device)
            deviation=models["lambda"]*abs(q-.5)/(1.+np.exp(-raw))
            prediction=median-deviation if q<.5 else median+deviation
        quantiles.append(prediction.T.astype(np.float32))
    return np.stack(quantiles,axis=-1)


def predict_stdk_target_series(base, models, centers, times, cfg, device, quantiles):
    """Return the fitted STDK series used as quantile-specific Q targets."""
    median=predict_spatial_stdk(models[.5],centers,times,cfg,device).T
    output={.5:median.astype(np.float32)}
    for q in quantiles:
        if q==.5:
            continue
        raw=predict_spatial_stdk(models[q],centers,times,cfg,device).T
        deviation=models["lambda"]*abs(q-.5)/(1.+np.exp(-raw))
        prediction=median-deviation if q<.5 else median+deviation
        output[q]=prediction.astype(np.float32)
    return output

def add_direct_stdk_baseline(prediction, stdk_models, centers, times, cfg, device):
    """Convert residual quantiles to final quantiles on standardized scale."""
    if cfg.prediction_mode != "residual":
        return prediction
    baseline=predict_spatial_stdk(
        stdk_models[.5],centers,np.asarray(times,dtype=np.int64),cfg,device
    ).T
    if prediction.shape[:2] != baseline.shape:
        raise ValueError(
            f"residual prediction shape {prediction.shape[:2]} != STDK baseline {baseline.shape}"
        )
    return prediction+baseline[...,None]

def fit_location(base, ds, q, cfg, device, median=None, validation_ds=None):
    if validation_ds is None:
        # Backward-compatible fallback for callers without an external Val block.
        nval_per_station=max(1,int(.05*ds.nw)); train_idx=[]; val_idx=[]
        for station in range(ds.y.shape[0]):
            begin=station*ds.nw; split=begin+ds.nw-nval_per_station; end=begin+ds.nw
            train_idx.extend(range(begin,split)); val_idx.extend(range(split,end))
        train=Subset(ds,train_idx); val=Subset(ds,val_idx)
    else:
        train=ds; val=validation_ds
    tl=DataLoader(train,batch_size=cfg.qconv_batch,shuffle=True); vl=DataLoader(val,batch_size=cfg.qconv_batch)
    bc=base.Config(seed=cfg.seed,train_times=cfg.train_times,horizon=cfg.horizon,
        lookback=cfg.lookback,grid_size=cfg.grid_size,qconv_epochs=cfg.qconv_epochs,
        qconv_batch_size=cfg.qconv_batch,qconv_lr=cfg.qconv_lr,
        qconv_patience=cfg.qconv_patience,conv_filters=cfg.conv_filters,
        qconv_profile="github_3block",quantile_lambda=ds.qlambda)
    model=base.build_qconvlstm(bc).to(device)
    opt=torch.optim.Adam(
        model.parameters(),lr=cfg.qconv_lr,weight_decay=cfg.qconv_weight_decay
    )
    if cfg.prediction_mode == "residual" and q == .5:
        # At initialization the combined median is exactly the direct STDK
        # baseline.  Training can retain that fallback or learn a correction.
        torch.nn.init.zeros_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
    if median is not None:
        median.eval()
        for p in median.parameters(): p.requires_grad_(False)
    selection_is_mse=cfg.prediction_mode == "residual" and q == .5

    def validation_selection_score():
        model.eval(); total=0.0; count=0; batch_losses=[]
        with torch.no_grad():
            for x,y,mx in vl:
                x=x.to(device); y=y.to(device); mx=mx.to(device)
                raw=model(x)
                pred=raw if median is None else base.constrained_quantile(raw,median(mx),q,ds.qlambda)
                if selection_is_mse:
                    total+=float(torch.nn.functional.mse_loss(pred,y,reduction="sum"))
                else:
                    batch_losses.append(float(base.pinball(pred,y,q)))
                count+=y.numel()
        return total/max(1,count) if selection_is_mse else float(np.mean(batch_losses))

    best=math.inf; state=None; wait=0
    if cfg.prediction_mode == "residual" and q == .5:
        # Include the zero-residual epoch-0 baseline in model selection, so
        # validation can reject every learned correction if none is helpful.
        best=validation_selection_score()
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"location residual QConv q={q:.2f} epoch=000 val_mse={best:.6f}")
    for epoch in range(1,cfg.qconv_epochs+1):
        model.train()
        for x,y,mx in tl:
            x,y,mx=x.to(device),y.to(device),mx.to(device); opt.zero_grad(set_to_none=True)
            raw=model(x); pred=raw if median is None else base.constrained_quantile(raw,median(mx),q,ds.qlambda)
            loss=base.pinball(pred,y,q); loss.backward(); opt.step()
        score=validation_selection_score()
        selection_label="mse" if selection_is_mse else "pinball"
        print(f"location QConv q={q:.2f} epoch={epoch:03d} val_{selection_label}={score:.6f}")
        if score < best-1e-6:
            best=score; wait=0; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=cfg.qconv_patience: break
    if state: model.load_state_dict(state)
    model.eval(); predictions=[]; truths=[]
    with torch.no_grad():
        for x,y,mx in vl:
            raw=model(x.to(device))
            pred=raw if median is None else base.constrained_quantile(
                raw,median(mx.to(device)),q,ds.qlambda
            )
            predictions.append(pred.cpu()); truths.append(y)
    val_pred=torch.cat(predictions); val_truth=torch.cat(truths)
    mse=float(torch.mean((val_pred-val_truth)**2))
    return model.eval(), {
        "validation_pinball":float(base.pinball(val_pred,val_truth,q)),
        "validation_rmse_scaled":math.sqrt(mse),
        "validation_mse_scaled":mse,
        "validation_mae_scaled":float(torch.mean(torch.abs(val_pred-val_truth))),
        "checkpoint_selection_loss":"mse" if selection_is_mse else "pinball",
        "best_checkpoint_selection_loss":best,
        "epoch0_zero_residual_candidate":cfg.prediction_mode == "residual" and q == .5,
        "training_windows":len(train),"validation_windows":len(val),
    }

def infer(base, models, frames, cfg, device, qlambda):
    parts=[]
    for start in range(0,len(frames[.5]),cfg.qconv_batch):
        stop=min(start+cfg.qconv_batch,len(frames[.5])); x={q:torch.from_numpy(frames[q][start:stop,-cfg.lookback:,None]).to(device) for q in frames}
        with torch.no_grad():
            med=models[.5](x[.5]); low=base.constrained_quantile(models[.05](x[.05]),med,.05,qlambda)
            high=base.constrained_quantile(models[.95](x[.95]),med,.95,qlambda)
        parts.append(torch.stack((low,med,high),-1).cpu().numpy())
    return np.concatenate(parts)


def predict_blocks(base, models, stdk_models, centers, target_start, target_length,
                   cfg, device, qlambda):
    """Predict contiguous blocks without using target-station observations.

    Each block receives five STDK-generated local grids immediately preceding
    its target origin. Callers must provide a target_start >= lookback; earlier
    boundary points use the direct-STDK warm-up rather than fictitious times.
    """
    if target_start < cfg.lookback:
        raise ValueError(
            f"target_start={target_start} has fewer than {cfg.lookback} "
            "legal history frames; use the direct-STDK boundary warm-up"
        )
    predictions=[]
    for start in range(target_start, target_start+target_length, cfg.horizon):
        history=np.arange(start-cfg.lookback,start)
        frames=frames_for(base,stdk_models,centers,history,cfg,device)
        block=infer(base,models,frames,cfg,device,qlambda)
        keep=min(cfg.horizon,target_start+target_length-start)
        predictions.append(block[:,:keep])
    return np.concatenate(predictions,axis=1)


def predict_blocks_cached(base, models, stdk_models, centers, target_start,
                          target_length, cfg, device, qlambda):
    """Block prediction with one batched STDK-grid build per scenario."""
    if target_start < cfg.lookback:
        raise ValueError(
            f"target_start={target_start} has fewer than {cfg.lookback} "
            "legal history frames; use the direct-STDK boundary warm-up"
        )
    windows,starts=block_history_windows(
        base,stdk_models,centers,target_start,target_length,cfg,device
    )
    prediction=infer(base,models,windows,cfg,device,qlambda)
    prediction=prediction.reshape(len(centers),len(starts),cfg.horizon,3)
    return prediction.reshape(len(centers),len(starts)*cfg.horizon,3)[:,:target_length]

def score(pred, truth):
    point = pred[..., 1]
    mse=float(np.mean((point-truth)**2))
    sse = float(np.sum((point-truth)**2))
    sst = float(np.sum((truth-np.mean(truth))**2))
    return {"RMSE":math.sqrt(mse),"MSE":mse,"MAE":float(np.mean(abs(pred[...,1]-truth))),
            "R2":float(1.0-sse/sst) if sst > 0 else float("nan"),
            "MPIW_90":float(np.mean(pred[...,2]-pred[...,0])),
            "coverage_90":float(np.mean((truth>=pred[...,0])&(truth<=pred[...,2])))}


def aggregate_location_validation(location_records, active_quantiles):
    """Aggregate independently fitted location models without pooling weights."""
    summary={}
    for q in active_quantiles:
        key=f"q{int(q*100):02d}"
        reports=[record["validation"][key] for record in location_records]
        weights=np.asarray([report["validation_windows"] for report in reports],dtype=float)
        weights=weights/weights.sum()
        mse=float(np.sum(weights*np.asarray([report["validation_mse_scaled"] for report in reports])))
        summary[key]={
            "validation_pinball":float(np.sum(weights*np.asarray([report["validation_pinball"] for report in reports]))),
            "validation_rmse_scaled":math.sqrt(mse),
            "validation_mse_scaled":mse,
            "validation_mae_scaled":float(np.sum(weights*np.asarray([report["validation_mae_scaled"] for report in reports]))),
            "checkpoint_selection_loss":reports[0]["checkpoint_selection_loss"],
            "mean_best_checkpoint_selection_loss":float(np.mean([report["best_checkpoint_selection_loss"] for report in reports])),
            "location_count":len(reports),
            "validation_windows":int(sum(report["validation_windows"] for report in reports)),
            "model_scope":"location_specific",
        }
    return summary


def run_seed(base, spatial, all_coords, all_values, cfg, output):
    began=time.time(); seed_all(cfg.seed)
    # Exact legacy RandomState sampling used by the existing Weather2K models.
    chosen=np.sort(np.random.RandomState(cfg.seed).choice(len(all_coords),cfg.n_sample,replace=False))
    held=np.sort(np.random.RandomState(cfg.seed+1).choice(cfg.n_sample,cfg.n_heldout,replace=False))
    train=np.setdiff1d(np.arange(cfg.n_sample),held)
    if len(train)!=cfg.n_train: raise AssertionError("not train500/heldout100")
    coords=norm_coords(all_coords[chosen]); values=all_values[chosen,-cfg.n_last:]
    obs=np.sort(np.random.RandomState(cfg.seed).choice(train,min(100,len(train)),replace=False))
    train_block=values[obs,:cfg.train_times]
    mean=float(train_block.mean()); std=float(train_block.std())
    if std < 1e-12: std=1.
    normalization={"source":cfg.normalization_source,"station_local":obs.tolist(),
                   "mean":mean,"std":std,"ddof":0,"train_times":cfg.train_times}
    # Do not even materialize a normalized held-out response array.  Only the
    # train500 block is transformed before final evaluation.
    scaled_train=((values[train]-mean)/std).astype(np.float32)
    targets=scaled_train[:,:cfg.train_times].T.reshape(-1)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qlambda=.5*float(targets.max()-targets.min())
    print(f"seed={cfg.seed} device={device} train={len(train)} heldout={len(held)}")
    train_times=np.arange(cfg.train_times,dtype=np.int64)
    val_times=np.arange(cfg.train_times,cfg.train_times+cfg.val_times,dtype=np.int64)
    med,median_stdk_validation=fit_spatial_stdk(
        base,spatial,coords[train],scaled_train,train_times,val_times,.5,cfg,device,qlambda
    )
    sm={.5:med,"lambda":qlambda}; stdk_validation={"q50":median_stdk_validation}
    active_quantiles=(.5,) if cfg.q50_only else (.05,.5,.95)
    for q in (.05,.95):
        if q in active_quantiles:
            sm[q],tail_validation=fit_spatial_stdk(
                base,spatial,coords[train],scaled_train,train_times,val_times,q,cfg,device,qlambda,med
            )
            stdk_validation[f"q{int(q*100):02d}"]=tail_validation
    target_by_scenario={
        "time_extrap_fixed500":("Target_Time150",train),
        "space_extrap_fixed850":("Target_Space100",held),
        "spatiotemp_100x150":("Target_ST100x150",held),
    }
    target_name,target_station_local=target_by_scenario[cfg.scenario]
    target_centers=coords[target_station_local]
    print(
        f"QConvLSTM scope=location_specific scenario={cfg.scenario} "
        f"target_locations={len(target_centers)}"
    )

    # Paper-style location-specific forecasting: each target s_0 receives its
    # own fitted-STDK series, local grids and independently optimized Q models.
    train_frames=frames_for(base,sm,target_centers,train_times,cfg,device)
    qconv_train_targets=predict_stdk_target_series(
        base,sm,target_centers,train_times,cfg,device,active_quantiles
    )
    if cfg.prediction_mode == "residual":
        train_stdk_baseline=predict_spatial_stdk(
            med,target_centers,train_times,cfg,device
        ).T
        qconv_train_targets={
            q:target-train_stdk_baseline for q,target in qconv_train_targets.items()
        }
    qconv_val_targets=predict_stdk_target_series(
        base,sm,target_centers,val_times,cfg,device,active_quantiles
    )
    if cfg.prediction_mode == "residual":
        val_stdk_baseline=predict_spatial_stdk(
            med,target_centers,val_times,cfg,device
        ).T
        qconv_val_targets={
            q:target-val_stdk_baseline for q,target in qconv_val_targets.items()
        }
    if cfg.forecast_mode == "block5to5":
        val_frames,val_starts=block_history_windows(
            base,sm,target_centers,cfg.train_times,cfg.val_times,cfg,device
        )
        val_targets={q:block_targets(qconv_val_targets[q],cfg.horizon) for q in active_quantiles}
        val_windows_per_location=len(val_starts)
    else:
        val_frames={q:train_frames[q][:,-cfg.lookback:] for q in active_quantiles}
        val_targets=qconv_val_targets
        val_windows_per_location=1

    truth_start=cfg.train_times+cfg.val_times
    space_length=truth_start
    location_validation=[]; prediction_parts=[]
    for position,station_local in enumerate(target_station_local):
        # Independent initialization and optimization for every target s_0.
        seed_all(cfg.seed*100000+int(station_local))
        location_slice=slice(position,position+1)
        datasets={q:SharedWindows(
            train_frames[q][location_slice],train_frames[.5][location_slice],
            qconv_train_targets[q][location_slice],cfg.lookback,cfg.horizon,qlambda
        ) for q in active_quantiles}
        val_slice=slice(
            position*val_windows_per_location,
            (position+1)*val_windows_per_location,
        )
        validation_sets={q:SharedValidation(
            val_frames[q][val_slice],val_frames[.5][val_slice],
            val_targets[q][val_slice],qlambda
        ) for q in active_quantiles}
        qm={}; validation={}
        qm[.5],validation[.5]=fit_location(
            base,datasets[.5],.5,cfg,device,validation_ds=validation_sets[.5]
        )
        for q in (.05,.95):
            if q in active_quantiles:
                qm[q],validation[q]=fit_location(
                    base,datasets[q],q,cfg,device,qm[.5],validation_sets[q]
                )
        location_validation.append({
            "station_local":int(station_local),
            "validation":{f"q{int(q*100):02d}":validation[q] for q in validation},
        })
        print(
            f"location-specific QConvLSTM {position+1}/{len(target_centers)} "
            f"station_local={int(station_local)}"
        )

        if not cfg.validation_only:
            center=target_centers[position:position+1]
            if cfg.scenario in {"time_extrap_fixed500","spatiotemp_100x150"}:
                if cfg.forecast_mode == "block5to5":
                    prediction=predict_blocks_cached(
                        base,qm,sm,center,truth_start,cfg.test_times,cfg,device,qlambda
                    )
                else:
                    history=np.arange(truth_start-cfg.lookback,truth_start)
                    prediction=infer(
                        base,qm,frames_for(base,sm,center,history,cfg,device),cfg,device,qlambda
                    )
                prediction=add_direct_stdk_baseline(
                    prediction,sm,center,
                    np.arange(truth_start,truth_start+cfg.test_times),cfg,device
                )
            else:
                warmup=min(cfg.lookback,space_length)
                direct=predict_stdk_quantiles(
                    base,sm,center,np.arange(warmup),cfg,device
                )
                remaining=space_length-warmup
                if remaining:
                    if cfg.forecast_mode == "block5to5":
                        forecast=predict_blocks_cached(
                            base,qm,sm,center,warmup,remaining,cfg,device,qlambda
                        )
                    else:
                        forecast=predict_blocks(
                            base,qm,sm,center,warmup,remaining,cfg,device,qlambda
                        )
                    forecast=add_direct_stdk_baseline(
                        forecast,sm,center,np.arange(warmup,space_length),cfg,device
                    )
                    prediction=np.concatenate((direct,forecast),axis=1)
                else:
                    prediction=direct
            prediction_parts.append(prediction[0])
        del qm

    validation=aggregate_location_validation(location_validation,active_quantiles)

    if cfg.validation_only:
        report={"model":model_name(cfg),"config":asdict(cfg),"normalization":normalization,
          "split":{"sampled_global":chosen.tolist(),"train_local":train.tolist(),
                   "heldout_local":held.tolist(),"heldout_truth_used_for_training":False,
                   "heldout_truth_used_for_selection":False},
          "selection_data":f"Per-location fitted-STDK Train700/chronological Val150 series on {len(target_centers)} target locations",
          "qconv_training_target":"quantile-specific fitted STDK series",
          "qconv_model_scope":"one independently fitted q05/q50/q95 QConvLSTM per target location",
          "prediction_definition":(
              "direct STDK q50 baseline + learned QConvLSTM residual quantiles"
              if cfg.prediction_mode == "residual" else
              "QConvLSTM direct target quantiles"
          ),
          "stdk_backend":"spatial_adapter","stdk_validation":stdk_validation,
          "validation":validation,"location_validation":location_validation,
          "tuning_quantiles":[float(q) for q in active_quantiles],
          "checkpoint_written":False,"elapsed_seconds":time.time()-began}
        output.mkdir(parents=True,exist_ok=True)
        stem=f"{output_stem(cfg)}_validation"
        (output/f"{stem}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report["validation"],indent=2))
        print("Validation-only run: Test150 and held-out100 metrics were not computed.")
        return report

    prediction=np.stack(prediction_parts)*std+mean
    if cfg.scenario == "time_extrap_fixed500":
        truth=values[train,truth_start:truth_start+cfg.test_times]
        stdk_prediction=predict_stdk_quantiles(
            base,sm,target_centers,np.arange(truth_start,truth_start+cfg.test_times),cfg,device
        )*std+mean
        output_suffix="time_forecasts.csv"; target_start=truth_start
        stdk_suffix="stdk_time_forecasts.csv"
    elif cfg.scenario == "space_extrap_fixed850":
        truth=values[held,:space_length]
        stdk_prediction=predict_stdk_quantiles(
            base,sm,target_centers,np.arange(space_length),cfg,device
        )*std+mean
        output_suffix="space_forecasts.csv"; target_start=0
        stdk_suffix="stdk_space_forecasts.csv"
    else:
        truth=values[held,truth_start:truth_start+cfg.test_times]
        stdk_prediction=predict_stdk_quantiles(
            base,sm,target_centers,np.arange(truth_start,truth_start+cfg.test_times),cfg,device
        )*std+mean
        output_suffix="forecasts.csv"; target_start=truth_start
        stdk_suffix="stdk_forecasts.csv"
    results={target_name:score(prediction,truth)}
    stdk_results={target_name:score(stdk_prediction,truth)}
    report={"model":model_name(cfg),"config":asdict(cfg),"normalization":normalization,
      "metrics":results[target_name],"results":results,
      "fitted_stdk_baseline":{
          "model":"Spatial-adapter STDK (same fitted models)",
          "refitted":False,"results":stdk_results,
          "normalization":normalization,
          "note":"Same fitted q05/q50/q95 STDK models used to generate QConvLSTM grids; not the historical obs100-scaled standalone run.",
      },
      "split":{"sampled_global":chosen.tolist(),"train_local":train.tolist(),"heldout_local":held.tolist(),
               "heldout_truth_used_for_training":False,"heldout_truth_role":"final_metrics_only"},
          "evaluation_protocol":{
          "training_scope":"independent QConvLSTM fit for every target location in this scenario",
          "scenario":cfg.scenario,
          "train":f"one quantile-specific fitted STDK Train700 series per target location ({len(target_centers)} independent fits)",
          "validation":"the same target location's fitted STDK Val150 series, checkpoint selection only",
          "Target_Time150":"Test150 x train500",
          "Target_Space100":"Fixed850 x held-out100",
          "Target_ST100x150":"Test150 x held-out100",
          "forecast_mode":cfg.forecast_mode,
          "prediction_mode":cfg.prediction_mode,
          "qconv_training_target":"quantile-specific fitted STDK series",
          "qconv_model_scope":"location_specific",
          "qconv_models_per_location":len(active_quantiles),
          "qconv_total_models":len(target_centers)*len(active_quantiles),
          "prediction_definition":(
              "direct Spatial-adapter STDK q50 at each target time plus learned QConvLSTM residual quantiles"
              if cfg.prediction_mode == "residual" else
              "QConvLSTM direct target quantiles"
          ),
          "median_residual_head_zero_initialized":cfg.prediction_mode == "residual",
          "space_block_forecast":f"direct {cfg.lookback}-to-{cfg.horizon} blocks",
          "space_boundary":(
              "times 0..4 use direct Spatial-adapter STDK warm-up; "
              "QConvLSTM starts at time 5 with legal STDK histories 0..4"
          ),
          "space_warmup_points":cfg.lookback,
          "heldout_truth_used_for_training":False,
      },
      "stdk_backend":"spatial_adapter","stdk_validation":stdk_validation,
      "validation":validation,"location_validation":location_validation,
      "checkpoint_written":False,"elapsed_seconds":time.time()-began}
    output.mkdir(parents=True,exist_ok=True)
    stem=output_stem(cfg)
    (output/f"{stem}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    def write_forecasts(path, station_indices, target_start, truth, prediction):
        with path.open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["seed","station_local","time_index","truth","q05","q50","q95"])
            for s,station_idx in enumerate(station_indices):
                for lead in range(truth.shape[1]):
                    w.writerow([cfg.seed,int(station_idx),target_start+lead,float(truth[s,lead]),*map(float,prediction[s,lead])])

    write_forecasts(
        output/f"{stem}_{output_suffix}",target_station_local,target_start,truth,prediction
    )
    write_forecasts(
        output/f"{stem}_{stdk_suffix}",target_station_local,target_start,truth,stdk_prediction
    )
    with (output/f"{stem}_comparison.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["seed","target","model",*next(iter(results.values())).keys()])
        for target in results:
            writer.writerow([cfg.seed,target,"fitted_stdk",*stdk_results[target].values()])
            writer.writerow([cfg.seed,target,model_name(cfg),*results[target].values()])
    print(json.dumps(results,indent=2)); print("No checkpoint was written."); return report

def parse_args():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument("--params-file",type=Path,default=BEST_PARAMS)
    known,_=pre.parse_known_args()
    tuned={}
    if known.params_file.exists():
        payload=json.loads(known.params_file.read_text(encoding="utf-8"))
        if (payload.get("stdk_backend")=="spatial_adapter"
            and payload.get("normalization_source")=="obs100_train700"
            and payload.get("stdk_q50_loss")=="pinball"
            and payload.get("qconv_training_target")=="quantile_specific_fitted_stdk"
            and payload.get("qconv_model_scope")=="location_specific"
            and payload.get("q50_checkpoint_selection")=="pinball"
            and payload.get("stdk_validation_aggregation")=="batch_mean"
            and payload.get("stdk_use_ema") is True):
            tuned=payload.get("formal_run_params",payload)
        else:
            print(
                f"Ignoring incompatible QConvLSTM params from {known.params_file}: "
                "expected spatial_adapter, obs100_train700 normalization, "
                "pinball q50, quantile-specific fitted-STDK targets, "
                "location-specific QConvLSTM models, "
                "batch-mean STDK validation and EMA."
            )
    p=argparse.ArgumentParser(description=__doc__,parents=[pre]); p.add_argument("--weather-data",type=Path,default=DATA)
    p.add_argument("--weather-variable",choices=VARIABLES,default="air_temperature"); p.add_argument("--seed",type=int,default=41)
    p.add_argument("--seeds",type=int,nargs="+"); p.add_argument("--output-dir",type=Path,default=OUTPUT)
    p.add_argument(
        "--scenario",
        choices=("time_extrap_fixed500","space_extrap_fixed850","spatiotemp_100x150","all_three"),
        default="all_three",
        help="Select the target locations for independent per-location QConvLSTM fits.",
    )
    p.add_argument(
        "--forecast-mode", choices=("block5to5", "direct5to150"),
        default="block5to5",
        help="Formal default is paper-aligned 5-to-5 blocks; direct5to150 is retained only for the archived pilot comparison.",
    )
    p.add_argument(
        "--prediction-mode",choices=("direct","residual"),default="direct",
        help="direct preserves the author-style head; residual adds its output to the direct Spatial-adapter STDK q50 forecast.",
    )
    p.add_argument("--stdk-epochs",type=int,default=350)
    p.add_argument("--grid-size",type=int,default=int(tuned.get("GRID_SIZE",8)))
    p.add_argument(
        "--neighbourhood-radius",type=float,
        default=float(tuned.get("NEIGHBOURHOOD_RADIUS",0.2)),
    )
    p.add_argument("--qconv-epochs",type=int,default=int(tuned.get("QCONV_EPOCHS",25)))
    p.add_argument("--qconv-batch-size",type=int,default=int(tuned.get("QCONV_BATCH_SIZE",64)))
    p.add_argument("--qconv-lr",type=float,default=float(tuned.get("QCONV_LR",1e-3)))
    p.add_argument(
        "--qconv-weight-decay",type=float,
        default=float(tuned.get("QCONV_WEIGHT_DECAY",0.0)),
    )
    p.add_argument("--qconv-patience",type=int,default=int(tuned.get("QCONV_PATIENCE",5)))
    p.add_argument("--conv-filters",type=int,default=int(tuned.get("CONV_FILTERS",64)))
    p.add_argument("--validation-only",action="store_true")
    p.add_argument(
        "--q50-only",action="store_true",
        help="Train only the median STDK/QConvLSTM path; valid only with --validation-only.",
    )
    p.add_argument("--smoke-test",action="store_true")
    args=p.parse_args()
    if not tuned and not (args.validation_only or args.smoke_test):
        p.error(
            "Formal evaluation requires compatible retuned params "
            "(obs100_train700, q50 pinball, fitted-STDK targets, "
            "location-specific QConvLSTM, batch-mean STDK validation, EMA). "
            "Run tuning first."
        )
    return args

def main():
    args=parse_args()
    if args.q50_only and not args.validation_only:
        raise SystemExit("--q50-only is a tuning shortcut and requires --validation-only")
    if args.scenario == "all_three":
        raise SystemExit(
            "Location-specific QConvLSTM requires one explicit --scenario to "
            "identify the target locations."
        )
    if args.grid_size < 1:
        raise SystemExit("--grid-size must be positive")
    if args.neighbourhood_radius <= 0:
        raise SystemExit("--neighbourhood-radius must be positive")
    if args.qconv_weight_decay < 0:
        raise SystemExit("--qconv-weight-decay cannot be negative")
    base=load_base(); spatial=load_spatial_stdk()
    coords,values=load_weather(args.weather_data.expanduser().resolve(),args.weather_variable)
    reports=[]
    for seed in args.seeds or [args.seed]:
        horizon=150 if args.forecast_mode == "direct5to150" else 5
        cfg=Config(seed=seed,variable=args.weather_variable,forecast_mode=args.forecast_mode,
                   prediction_mode=args.prediction_mode,scenario=args.scenario,horizon=horizon,stdk_epochs=args.stdk_epochs,
                   qconv_epochs=args.qconv_epochs,qconv_batch=args.qconv_batch_size,
                   qconv_lr=args.qconv_lr,qconv_weight_decay=args.qconv_weight_decay,
                   qconv_patience=args.qconv_patience,conv_filters=args.conv_filters,
                   grid_size=args.grid_size,radius=args.neighbourhood_radius,
                   validation_only=args.validation_only,q50_only=args.q50_only)
        if args.smoke_test:
            cfg.n_sample,cfg.n_train,cfg.n_heldout=16,12,4; cfg.n_last,cfg.train_times,cfg.val_times,cfg.test_times=50,30,10,10
            cfg.horizon=10 if cfg.forecast_mode == "direct5to150" else 5
            cfg.stdk_epochs,cfg.qconv_epochs,cfg.qconv_batch=1,1,8; cfg.smoke=True
        reports.append(run_seed(base,spatial,coords,values,cfg,args.output_dir))
    if len(reports)>1 and not args.validation_only:
        summary={
            "model":model_name(cfg),
            "seed_list":[int(r["config"]["seed"]) for r in reports],
            "results":{
                target:{
                    metric:{
                        "mean":float(np.mean([r["results"][target][metric] for r in reports])),
                        "std":float(np.std([r["results"][target][metric] for r in reports],ddof=0)),
                    }
                    for metric in reports[0]["results"][target]
                }
                for target in reports[0]["results"]
            },
        }
        summary["forecast_mode"]=args.forecast_mode
        summary["prediction_mode"]=args.prediction_mode
        summary_seeds=[int(r["config"]["seed"]) for r in reports]
        seed_label=(
            "five_seed"
            if summary_seeds == [41,42,43,44,45]
            else "seeds" + "_".join(str(seed) for seed in summary_seeds)
        )
        scenario_label="" if args.scenario == "all_three" else f"_{args.scenario}"
        summary_name=f"{output_prefix(cfg)}_{args.forecast_mode}{scenario_label}_{seed_label}_summary.json"
        (args.output_dir/summary_name).write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
