from generate_sample import generate_fake_sonar_sweep
from filter import denoise, detect_targets

raw_signal = generate_fake_sonar_sweep()
clean_signal = denoise(raw_signal)
targets = detect_targets(clean_signal)

print(f"Raw signal length: {len(raw_signal)}")
print(f"Detected {len(targets)} target(s):")
for t in targets:
    print(t)