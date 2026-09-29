from __future__ import annotations
import sys, time, math, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.linalg import eigh

ROOT = Path('/mnt/data/SCQM/SCQM/SCQM_Three_Experiments_Unified')
import importlib

def _load_local(module_name, path):
    oldpath=list(sys.path)
    sys.path.insert(0, str(path))
    try:
        mod=importlib.import_module(module_name)
    finally:
        sys.path[:] = oldpath
    return mod

syn=_load_local('run_tableii_synthetic', ROOT/'experiments/synthetic/src')
# Clear local helper module names so the Bunny code imports its own copies.
for _m in ['methods','scqm_njit','real_method_grid','stats_utils']:
    sys.modules.pop(_m,None)
bunny=_load_local('run_bunny', ROOT/'experiments/bunny/src')
for _m in ['methods','scqm_njit','real_method_grid','stats_utils']:
    sys.modules.pop(_m,None)
coil=_load_local('run_coil20_exactgrid', ROOT/'experiments/coil20/src')

OUT=Path('/mnt/data/mfit_lkde_results')
OUT.mkdir(exist_ok=True)


def _pca_reduce_if_needed(Data_rows: np.ndarray, points_rows: np.ndarray, d: int, threshold: int=64):
    """Exact isometric reduction to the affine span for very high-D small-n data."""
    X=np.asarray(Data_rows,float); P=np.asarray(points_rows,float)
    n,D=X.shape
    if D <= threshold:
        return X,P,None
    mu=X.mean(0)
    Y=X-mu
    # rank at most n-1; full_matrices=False is cheap for n << D
    _,s,vt=np.linalg.svd(Y, full_matrices=False)
    tol=max(Y.shape)*np.finfo(float).eps*(s[0] if len(s) else 1.0)
    q=max(d+1, int(np.sum(s>tol)))
    q=min(q, vt.shape[0])
    W=vt[:q].T
    return Y@W, (P-mu)@W, (mu,W)


def _lift(Y: np.ndarray, lift):
    if lift is None: return Y
    mu,W=lift
    return mu + np.asarray(Y)@W.T


def _knn_sigma(x, X, k):
    d2=np.sum((X-x)**2, axis=1)
    ind=np.argsort(d2)
    kk=min(int(k), len(X)-1)
    if kk < 1: return 1e-12, ind
    # MATLAB: ind(2:k+1)
    chosen=ind[1:kk+1]
    sigma=float(np.max(np.sqrt(d2[chosen]))) if len(chosen) else float(np.sqrt(d2[ind[-1]]))
    return max(sigma,1e-12), ind


def log_kde_project(Data_rows, point, d, k, tol=1e-6, max_iter=49):
    X0=np.asarray(Data_rows,float); p0=np.asarray(point,float).reshape(1,-1)
    X,P,lift=_pca_reduce_if_needed(X0,p0,d)
    x0=P[0].copy(); old=np.zeros_like(x0); move=x0.copy(); sk=1
    while np.linalg.norm(old-move)>tol and sk<50:
        old=move.copy()
        h, ind=_knn_sigma(move,X,k)
        kk=min(int(k),len(X)-1)
        # MATLAB shift_mean uses first k sorted entries, which includes nearest/current point.
        use=ind[:max(1,kk)]
        dist2=np.sum((X[use]-move)**2,axis=1)
        w=np.exp(-dist2/(h*h))
        sw=float(w.sum())
        mean=(w[:,None]*X[use]).sum(0)/max(sw,1e-300)
        # fitting(mean, Data, h): weighted covariance about shifted mean, all data.
        A=X-mean
        wa=np.exp(-np.sum(A*A,axis=1)/(h*h))
        C=A.T@(wa[:,None]*A)
        # top d eigenvectors of PSD C
        vals,vecs=np.linalg.eigh(C)
        Uh=vecs[:,np.argsort(vals)[::-1][:d]]
        delta=mean-old
        move=old + delta - Uh@(Uh.T@delta)
        sk += 1
    return _lift(move.reshape(1,-1),lift)[0]


def _normal_projector_local(X, i, r2, d, pair_d2=None):
    if pair_d2 is None:
        dd=np.sum((X-X[i])**2,axis=1)
    else:
        dd=pair_d2[i]
    mask=(dd>0)&(dd<=r2)
    A=X[mask]-X[i]
    D=X.shape[1]
    if len(A)==0:
        # MATLAB svd(zeros) gives an arbitrary orthonormal basis; use coordinate basis.
        Uh=np.eye(D)[:,:d]
    else:
        C=A.T@A
        vals,vecs=np.linalg.eigh(C)
        Uh=vecs[:,np.argsort(vals)[::-1][:d]]
    return np.eye(D)-Uh@Uh.T


def mfit_project(Data_rows, point, d, k, tol=1e-6):
    X0=np.asarray(Data_rows,float); p0=np.asarray(point,float).reshape(1,-1)
    X,P,lift=_pca_reduce_if_needed(X0,p0,d)
    move=P[0].copy(); old=np.zeros_like(move); sk=1
    D=X.shape[1]
    # For small data, pairwise distances are fastest; for Bunny-sized data,
    # use a KD-tree to avoid allocating an n x n matrix for every query.
    pair_d2=None if len(X)>500 else np.sum((X[:,None,:]-X[None,:,:])**2,axis=2)
    tree=cKDTree(X) if pair_d2 is None else None
    beta=0.1; step=0.5
    while np.linalg.norm(old-move)>tol and sk<50:
        old=move.copy()
        r,_=_knn_sigma(move,X,k)
        r2=r*r
        dist2=np.sum((X-move)**2,axis=1)
        d2=1.0-dist2/r2
        d2[d2<0]=0.0
        at=np.power(d2,beta,where=(d2>=0),out=np.zeros_like(d2))
        asum=float(at.sum())
        if not np.isfinite(asum) or asum<=1e-300:
            break
        alpha=at/asum
        active=np.flatnonzero(d2>0)
        Ns=np.zeros((D,D),float); cvec=np.zeros(D,float)
        for i in active:
            ii=int(i)
            if pair_d2 is not None:
                Pi=_normal_projector_local(X,ii,r2,d,pair_d2)
            else:
                nbr=np.asarray(tree.query_ball_point(X[ii], r), dtype=int)
                if len(nbr):
                    dd=np.sum((X[nbr]-X[ii])**2,axis=1)
                    nbr=nbr[dd>0]
                A=X[nbr]-X[ii] if len(nbr) else np.empty((0,D))
                if len(A)==0:
                    Uh=np.eye(D)[:,:d]
                else:
                    C=A.T@A; vals0,vecs0=np.linalg.eigh(C)
                    Uh=vecs0[:,np.argsort(vals0)[::-1][:d]]
                Pi=np.eye(D)-Uh@Uh.T
            a=float(alpha[ii])
            Ns += a*Pi
            cvec += a*(Pi@(X[ii]-move))
        vals,vecs=np.linalg.eigh(Ns)
        # MATLAB U(:,1:D-d) from SVD descending singular values = normal projector.
        idx=np.argsort(vals)[::-1][:max(0,D-d)]
        if len(idx):
            Un=vecs[:,idx]; Pn=Un@Un.T
        else:
            Pn=np.zeros((D,D))
        direction=step*(Pn@cvec)  # inner mfit multiplies by step
        move=old + step*direction # caller multiplies by step again
        sk += 1
    return _lift(move.reshape(1,-1),lift)[0]


def synthetic_run():
    rows=[]
    sigmas=(0.03,0.09,0.15); seed=43; n=220; k_default=36; n_eval=12
    kmap={'quadratic10d':90}
    for manifold in syn.MANIFOLDS:
      for noise in syn.NOISES:
       for sigma in sigmas:
        rng=np.random.default_rng(seed)
        clean,_=syn.generate_manifold(manifold,n,rng)
        noisy=clean+sigma*syn.make_noise(noise,clean.shape,rng)
        eval_ids=np.linspace(0,n-1,n_eval,dtype=int)
        k=int(kmap.get(manifold,k_default))
        for target in eval_ids:
          for method,fn in [('MFIT',mfit_project),('LOG-KDE',log_kde_project)]:
            t=time.perf_counter()
            try:
              est=fn(noisy,noisy[target],2,k)
              err=float(np.sum((est-clean[target])**2)); success=np.isfinite(err); msg=''
            except Exception as e:
              err=np.nan;success=False;msg=repr(e)
            rows.append(dict(manifold=manifold,noise=noise,sigma=sigma,seed=seed,target=int(target),method=method,squared_error=err,runtime_s=time.perf_counter()-t,success=success,error=msg,k=k))
        print('synthetic',manifold,noise,sigma,flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'synthetic_samples.csv',index=False)
    summ=df.groupby(['manifold','noise','sigma','method'],as_index=False).agg(mean_sse=('squared_error','mean'),se_sse=('squared_error',lambda x:x.std(ddof=1)/math.sqrt(max(x.notna().sum(),1))),n=('squared_error','count'),mean_runtime_s=('runtime_s','mean'))
    summ.to_csv(OUT/'synthetic_summary.csv',index=False)
    return df,summ


def bunny_run():
    mesh=ROOT/'datasets/bunny/bun_zipper.ply'
    rows=[]; seed=43;n_points=2000;k=50;n_eval=30;ref_points=120000
    for noise in bunny.NOISES:
      for sigma in bunny.SIGMAS:
        clean,clean_normals,reference,reference_normals=bunny.sample_reference(str(mesh),n_points,ref_points,sample_seed=seed)
        rng=np.random.default_rng(seed); noisy=clean+sigma*bunny.generate_noise(noise,clean.shape,rng)
        ref_tree=cKDTree(reference); eval_ids=np.linspace(0,n_points-1,n_eval,dtype=int)
        for target in eval_ids:
          for method,fn in [('MFIT',mfit_project),('LOG-KDE',log_kde_project)]:
            t=time.perf_counter()
            try:
              est=fn(noisy,noisy[target],2,k)
              corr=float(np.sum((est-clean[target])**2)); dist,near=ref_tree.query(est); surf=float(dist*dist); success=np.isfinite(surf);msg=''
            except Exception as e:
              corr=surf=np.nan;success=False;msg=repr(e)
            rows.append(dict(dataset='Stanford Bunny',noise=noise,sigma=float(sigma),seed=seed,target=int(target),method=method,corresponding_sse=corr,surface_sse=surf,runtime_s=time.perf_counter()-t,success=success,error=msg,k=k))
        print('bunny',noise,sigma,flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'bunny_samples.csv',index=False)
    summ=df.groupby(['noise','sigma','method'],as_index=False).agg(mean_surface_sse=('surface_sse','mean'),se_surface_sse=('surface_sse',lambda x:x.std(ddof=1)/math.sqrt(max(x.notna().sum(),1))),mean_corresponding_sse=('corresponding_sse','mean'),n=('surface_sse','count'),mean_runtime_s=('runtime_s','mean'))
    summ.to_csv(OUT/'bunny_summary.csv',index=False)
    return df,summ


def coil_run():
    coil_dir=Path('/mnt/data/coil20_data/coil-20-proc')
    views=[0,2,4,6,8]; angles=[0,10,20,30,40]
    paths=coil.selected_paths(coil_dir,20,views); groups=coil.load_groups(paths)
    rows=[]
    for obj in sorted(groups):
        imgs=groups[obj]; H,W=imgs.shape[1:]; X=imgs.reshape(len(imgs),-1)
        k=min(4,len(X)-1)
        for method,fn in [('MFIT',mfit_project),('LOG-KDE',log_kde_project)]:
            t0=time.perf_counter(); rec=[]
            for j in range(len(X)):
                rec.append(fn(X,X[j],2,k))
            runtime=time.perf_counter()-t0
            rec=np.clip(np.asarray(rec).reshape(len(imgs),H,W),0.0,1.0)
            for j in range(len(imgs)):
                met=coil.image_metrics(rec[j],imgs[j])
                rows.append(dict(object=obj,view_index=views[j],angle_deg=angles[j],method=method,runtime_s=runtime,**met))
        print('coil',obj,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'coil20_samples.csv',index=False)
    objm=df.groupby(['object','method'],as_index=False).agg(mse=('mse','mean'),mae=('mae','mean'),psnr=('psnr','mean'),ssim=('ssim','mean'))
    objm.to_csv(OUT/'coil20_object_summary.csv',index=False)
    summ=df.groupby('method',as_index=False).agg(mean_mse=('mse','mean'),mean_mae=('mae','mean'),mean_psnr=('psnr','mean'),mean_ssim=('ssim','mean'),mean_runtime_s=('runtime_s','mean'))
    summ.to_csv(OUT/'coil20_summary.csv',index=False)
    return df,summ

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['synthetic','bunny','coil','all'],default='all');args=ap.parse_args()
    if args.part in ('synthetic','all'): synthetic_run()
    if args.part in ('bunny','all'): bunny_run()
    if args.part in ('coil','all'): coil_run()
