import numpy as np
import pandas as pd
import pickle, os
from data_preprocessing import Preprocessor

# ---------- utils ----------
def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

def kfold_splits(n, k=5, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val

def metrics_block(name, y, yhat):
    mse = np.mean((y - yhat)**2)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y, yhat)
    print("\nRegression Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²) Score: {r2:.2f}")
    print(f"({name})")

# ---------- model fits (return dicts) ----------
def ols_fit(X, y):
    XtX, Xty = X.T @ X, X.T @ y
    try: w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError: w = np.linalg.pinv(X) @ y
    return {"type": "ols", "w": w}

def ridge_fit(X, y, alpha):
    n = X.shape[1]
    I = np.eye(n); I[0,0] = 0.0  # don't penalize bias
    w = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
    return {"type": "ridge", "alpha": float(alpha), "w": w}

def lasso_fit(X, y, alpha, max_iter=2000, tol=1e-4):
    n, d = X.shape
    w = np.zeros(d); col_sq = (X**2).sum(axis=0) + 1e-12
    yhat = X @ w
    def soft(rho, lam): 
        return np.sign(rho)*max(abs(rho)-lam, 0.0)
    for _ in range(max_iter):
        w_old = w.copy()
        # intercept j=0 (no penalty)
        r = y - yhat + w[0]*X[:,0]
        w0 = (X[:,0] @ r) / col_sq[0]
        yhat += (w0 - w[0]) * X[:,0]; w[0] = w0
        for j in range(1, d):
            r = y - yhat + w[j]*X[:,j]
            rho = X[:,j] @ r
            wj = soft(rho, alpha) / col_sq[j]
            yhat += (wj - w[j]) * X[:,j]; w[j] = wj
        if np.max(np.abs(w - w_old)) < tol: break
    return {"type": "lasso", "alpha": float(alpha), "w": w}

def enet_fit(X, y, alpha, l1_ratio, max_iter=2000, tol=1e-4):
    n, d = X.shape
    w = np.zeros(d); col_sq = (X**2).sum(axis=0) + 1e-12
    yhat = X @ w
    l1 = alpha*l1_ratio; l2 = alpha*(1.0-l1_ratio)
    def soft(rho, lam): 
        return np.sign(rho)*max(abs(rho)-lam, 0.0)
    for _ in range(max_iter):
        w_old = w.copy()
        r = y - yhat + w[0]*X[:,0]
        w0 = (X[:,0] @ r) / col_sq[0]
        yhat += (w0 - w[0]) * X[:,0]; w[0] = w0
        for j in range(1, d):
            r = y - yhat + w[j]*X[:,j]
            rho = X[:,j] @ r
            denom = col_sq[j] + l2
            wj = soft(rho, l1) / denom
            yhat += (wj - w[j]) * X[:,j]; w[j] = wj
        if np.max(np.abs(w - w_old)) < tol: break
    return {"type": "elasticnet", "alpha": float(alpha), "l1_ratio": float(l1_ratio), "w": w}

def poly_expand(X, degree=2):
    bias, F = X[:,[0]], X[:,1:]
    parts = [F] + [F**d for d in range(2, degree+1)]
    return np.hstack([bias] + parts)

def polyols_fit(X, y, degree=2):
    Xp = poly_expand(X, degree)
    try: w = np.linalg.solve(Xp.T @ Xp, Xp.T @ y)
    except np.linalg.LinAlgError: w = np.linalg.pinv(Xp) @ y
    return {"type": "poly_ols", "degree": int(degree), "w": w}

def polyridge_fit(X, y, degree, alpha):
    Xp = poly_expand(X, degree)
    n = Xp.shape[1]
    I = np.eye(n); I[0,0] = 0.0
    w = np.linalg.solve(Xp.T @ Xp + alpha*I, Xp.T @ y)
    return {"type": "poly_ridge", "degree": int(degree), "alpha": float(alpha), "w": w}

# ---------- inference for dict models ----------
def predict_from_dict(model, X):
    t = model["type"]
    if t == "ols" or t == "ridge" or t == "lasso" or t == "elasticnet":
        return X @ model["w"]
    elif t == "poly_ols":
        return poly_expand(X, model["degree"]) @ model["w"]
    elif t == "poly_ridge":
        return poly_expand(X, model["degree"]) @ model["w"]
    else:
        raise ValueError(f"Unknown model type: {t}")

# ---------- CV ----------
def mean_cv_r2(X, y, ctor):
    scores=[]
    for tr, va in kfold_splits(len(y), k=5, seed=42):
        m = ctor()
        yhat = predict_from_dict(m, X[va]) if isinstance(m, dict) else None  # not used
        # rebuild via ctor each fold
        model = ctor()
        yhat = predict_from_dict(model, X[va]) if False else None  # placeholder
        # better: ctor returns a function that fits and returns model dict
    return 0.0
def cv_for_family(X, y, family):
    """Return (best_model_dict, best_cv_r2). family is a string."""
    if family == "ols":
        m = ols_fit(X, y)
        # OLS has no hyperparam; approximate CV by training each fold and averaging
        scores=[]
        for tr, va in kfold_splits(len(y), 5, 42):
            m_fold = ols_fit(X[tr], y[tr])
            scores.append(r2_score(y[va], predict_from_dict(m_fold, X[va])))
        return m, float(np.mean(scores))

    if family == "ridge":
        grid = [0.1, 0.3, 1.0, 3.0, 10.0]
        best, best_r2 = None, -np.inf
        for a in grid:
            scores=[]
            for tr, va in kfold_splits(len(y), 5, 42):
                m_fold = ridge_fit(X[tr], y[tr], a)
                scores.append(r2_score(y[va], predict_from_dict(m_fold, X[va])))
            r2 = float(np.mean(scores))
            if r2 > best_r2:
                best_r2, best = r2, ridge_fit(X, y, a)
        return best, best_r2

    if family == "lasso":
        grid = [0.0005, 0.001, 0.003, 0.01, 0.03]
        best, best_r2 = None, -np.inf
        for a in grid:
            scores=[]
            for tr, va in kfold_splits(len(y), 5, 42):
                m_fold = lasso_fit(X[tr], y[tr], a)
                scores.append(r2_score(y[va], predict_from_dict(m_fold, X[va])))
            r2 = float(np.mean(scores))
            if r2 > best_r2:
                best_r2, best = r2, lasso_fit(X, y, a)
        return best, best_r2

    if family == "elasticnet":
        a_grid = [0.001, 0.003, 0.01, 0.03, 0.1]
        t_grid = [0.2, 0.5, 0.8]
        best, best_r2 = None, -np.inf
        for a in a_grid:
            for t in t_grid:
                scores=[]
                for tr, va in kfold_splits(len(y), 5, 42):
                    m_fold = enet_fit(X[tr], y[tr], a, t)
                    scores.append(r2_score(y[va], predict_from_dict(m_fold, X[va])))
                r2 = float(np.mean(scores))
                if r2 > best_r2:
                    best_r2, best = r2, enet_fit(X, y, a, t)
        return best, best_r2

    if family == "poly_ols":
        # degree fixed at 2 per your request
        scores=[]
        for tr, va in kfold_splits(len(y), 5, 42):
            m_fold = polyols_fit(X[tr], y[tr], degree=2)
            scores.append(r2_score(y[va], predict_from_dict(m_fold, X[va])))
        return polyols_fit(X, y, degree=2), float(np.mean(scores))

    if family == "poly_ridge":
        grid = [0.1, 0.3, 1.0, 3.0, 10.0]
        best, best_r2 = None, -np.inf
        for a in grid:
            scores=[]
            for tr, va in kfold_splits(len(y), 5, 42):
                m_fold = polyridge_fit(X[tr], y[tr], degree=2, alpha=a)
                scores.append(r2_score(y[va], predict_from_dict(m_fold, X[va])))
            r2 = float(np.mean(scores))
            if r2 > best_r2:
                best_r2, best = r2, polyridge_fit(X, y, degree=2, alpha=a)
        return best, best_r2

    raise ValueError("unknown family")
def main():
    # 1) data → X,y
    df = pd.read_csv("data/train_data.csv")
    pre = Preprocessor(target_col="life_expectancy")
    X, y = pre.fit_transform(df)

    # 2) train families with 5-fold CV
    families = ["ols", "ridge", "lasso", "elasticnet", "poly_ols", "poly_ridge"]
    results = []
    for fam in families:
        model, cv_r2 = cv_for_family(X, y, fam)
        results.append((fam, model, cv_r2))

    # 3) rank and save top-3 + final
    results.sort(key=lambda x: x[2], reverse=True)
    os.makedirs("models", exist_ok=True)
    for i, (fam, model, _) in enumerate(results[:3], start=1):
        pickle.dump({"model": model, "preprocessor": pre}, open(f"models/regression_model{i}.pkl", "wb"))
    final_fam, final_model, final_cv = results[0]
    pickle.dump({"model": final_model, "preprocessor": pre}, open("models/regression_model_final.pkl", "wb"))
    print(f"Saved models to /models | final={final_fam} (CV R²={final_cv:.3f})")

    # 4) print metrics blocks on full training set (your exact format)
    for fam, model, _ in results[:3] + [("final", final_model, final_cv)]:
        yhat = predict_from_dict(model, X)
        metrics_block(fam, y, yhat)

if __name__ == "__main__":
    main()



