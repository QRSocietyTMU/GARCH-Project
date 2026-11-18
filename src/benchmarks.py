def hist_var(window):
  return lambda train: train.iloc[-window:].var()

def ewma(lam):
  return lambda train: (train ** 2).ewm(alpha=1-lam, adjust=False).mean().iloc[-1]