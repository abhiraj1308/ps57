import numpy as np

def generate_fake_sonar_sweep(num_points=200, noise_level=0.05):
    """Simulates one sonar sweep: signal intensity at each range step.
    Injects two fake 'objects' as intensity plateaus (a few points wide,
    like a real object would actually appear across several range bins)."""
    signal = np.random.normal(0, noise_level, num_points)
    signal[48:53] += 0.8    # fake object 1 — spans 5 points now, survives smoothing
    signal[138:143] += 0.6  # fake object 2
    return signal