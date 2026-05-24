import numpy as np


class MarketRegimeDetector:
    """Определяет тип рынка по трём метрикам."""

    def __init__(self,
                 trend_up=0.55,
                 trend_down=0.45,
                 vol_ratio_threshold=1.3,
                 autocorr_threshold=0.05):
        self.trend_up = trend_up
        self.trend_down = trend_down
        self.vol_ratio_threshold = vol_ratio_threshold
        self.autocorr_threshold = autocorr_threshold

    def detect(self, returns: np.ndarray) -> dict:
        clean = returns[~np.isnan(returns)]

        if len(clean) < 20:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'metrics': {}
            }

        prob_up = float((clean > 0).mean())
        trend_strength = abs(prob_up - 0.5) * 2

        vol_short = np.nanstd(clean[-10:]) if len(clean) >= 10 else np.nanstd(clean)
        vol_long = np.nanstd(clean) if len(clean) > 0 else 1e-10
        vol_ratio = vol_short / vol_long

        if len(clean) > 2:
            autocorr = np.corrcoef(clean[:-1], clean[1:])[0, 1]
            autocorr = 0.0 if np.isnan(autocorr) else autocorr
        else:
            autocorr = 0.0

        if vol_ratio > self.vol_ratio_threshold:
            regime = 'volatile'
            confidence = min(1.0, (vol_ratio - 1.0))
        elif prob_up > self.trend_up:
            regime = 'trend_up'
            confidence = trend_strength
        elif prob_up < self.trend_down:
            regime = 'trend_down'
            confidence = trend_strength
        else:
            regime = 'sideways'
            confidence = 1.0 - trend_strength

        return {
            'regime': regime,
            'confidence': float(confidence),
            'metrics': {
                'prob_up': float(prob_up),
                'trend_strength': float(trend_strength),
                'vol_ratio': float(vol_ratio),
                'autocorr': float(autocorr),
            }
        }