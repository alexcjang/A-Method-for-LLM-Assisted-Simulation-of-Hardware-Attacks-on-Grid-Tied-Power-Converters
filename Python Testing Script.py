import csv
import math
import time
from itertools import zip_longest

import rtds.rscadfx
from rtds.constants import SLIDER

# Test configuration
mag_values = [0.0, 0.01, 0.05, 0.07, 0.12, 0.2]
attack_type = "3phase_magnitude_corruption"

# Change these for each generated/reference file.
trial_id = "3.11"
evaluation_stage = "original"  # Or "after_one_manual_fix"

# Peak window: at least two 60 Hz cycles.
# This must be the verified WAVEFORM RECORDING interval,
# not merely the simulation time step.
waveform_sample_interval = 50e-6
grid_frequency = 60.0
peak_window_samples = (
    math.ceil((2.0 / grid_frequency) / waveform_sample_interval) + 1
)

# Preserve the original RMS-summary calculation.
rms_window_samples = 50

output_prefix = f"{trial_id}_{evaluation_stage}"

# Connect to RSCAD FX
with rtds.rscadfx.remote_connection() as app:
    case = app.open_case("Microgrid1_GPT_test3.11.rtfx")

    # Retrieve slider handle
    atk_slider = case.get_object_by_name("ATK_MAG_3phase", SLIDER)

    # Retrieve signal handles
    sig_prt8 = case.get_signal(
        "Subsystem #1|CTLs|Vars|PROTECT8"
    )
    sig_vrms_meas = case.get_signal(
        "Subsystem #1|CTLs|Vars|VPROT8"
    )
    sig_vrms_real = case.get_signal(
        "Subsystem #1|CTLs|Vars|VLOADRMS8_Real"
    )
    sig_vpeak_real = case.get_signal(
        "Subsystem #1|Node Voltages|S1) VSAA8"
    )

    # Fail explicitly if a required object is missing.
    required_objects = {
        "ATK_MAG_3phase": atk_slider,
        "PROTECT8": sig_prt8,
        "VPROT8": sig_vrms_meas,
        "VLOADRMS8_Real": sig_vrms_real,
        "VSAA8": sig_vpeak_real,
    }

    for name, obj in required_objects.items():
        if obj is None:
            raise RuntimeError(f"Required object not found: {name}")

    with open(
        f"{output_prefix}_summary_results.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Trial_ID",
            "Evaluation_Stage",
            "ATK_MAG",
            "Attack_Type",
            "V_RMS_Real_kV",
            "VPROT_Mean_pu",
            "PhaseA_MaxAbs_Final2Cycles_kV",
            "Protection_High_In_Buffer",
            "Trip_Time_s",
            "Trip_Time_Status",
            "Data_Status",
        ])

        for val in mag_values:
            try:
                # 1. Start with the attack disabled.
                # Retains your verified fresh-start behavior between runs.
                atk_slider.value = 0.0
                case.run()

                settle_time = 2.0
                time.sleep(settle_time)

                # 2. Introduce attack; 0.0 is the attack-disabled test.
                atk_slider.value = val
                app.print_message(
                    f"Testing ATK_MAG_3phase = {val} | "
                    f"Type: {attack_type}"
                )

                # 3. Allow post-attack operation.
                test_duration = 4.0
                time.sleep(test_duration)

                # Fetch plot buffer arrays.
                # Capture freshness and alignment must be verified in Runtime.
                case.update_plots()

                prt8_data = list(sig_prt8.get_data())
                vrms_meas_data = list(sig_vrms_meas.get_data())
                vrms_real_data = list(sig_vrms_real.get_data())
                vpeak_real_data = list(sig_vpeak_real.get_data())

                # Save the complete retrieved buffers for inspection.
                # Rows are buffer indices, NOT verified synchronized times.
                # Unequal buffer lengths are padded with blank cells.
                with open(
                    f"{output_prefix}_raw_{val:.3f}kV.csv",
                    "w",
                    newline="",
                ) as rawfile:
                    raw_writer = csv.writer(rawfile)
                    raw_writer.writerow([
                        "Buffer_Index",
                        "PROTECT8",
                        "VPROT8_pu",
                        "VLOADRMS8_Real_kV",
                        "VSAA8_kV",
                    ])

                    for idx, values in enumerate(zip_longest(
                        prt8_data,
                        vrms_meas_data,
                        vrms_real_data,
                        vpeak_real_data,
                        fillvalue="",
                    )):
                        raw_writer.writerow([idx, *values])

                # Reject missing/invalid data instead of replacing with zero.
                checks = [
                    ("PROTECT8", prt8_data, 1),
                    ("VPROT8", vrms_meas_data, rms_window_samples),
                    ("VLOADRMS8_Real", vrms_real_data, rms_window_samples),
                    ("VSAA8", vpeak_real_data, peak_window_samples),
                ]

                for name, data, minimum_count in checks:
                    if len(data) < minimum_count:
                        raise ValueError(
                            f"{name}: requires {minimum_count} samples, "
                            f"received {len(data)}"
                        )

                    if not all(math.isfinite(float(x)) for x in data):
                        raise ValueError(
                            f"{name}: contains nonfinite values"
                        )

                # --- Calculation 1: Protection status ---
                # Reports whether a high flag was present in the buffer.
                # A high first sample does NOT establish a startup trip.
                tripped = any(x > 0.5 for x in prt8_data)

                # Accurate delay requires synchronized attack-onset and
                # protection timestamps. Do not infer it from buffer length.
                trip_time = ""
                trip_time_status = (
                    "not_applicable_zero_bias"
                    if val == 0.0
                    else "unavailable_without_event_timestamps"
                )

                # --- Calculation 2: Real RMS and protection-input means ---
                # Store full precision; round only for display if desired.
                real_window = vrms_real_data[-rms_window_samples:]
                measured_window = vrms_meas_data[-rms_window_samples:]

                v_rms_real = sum(real_window) / len(real_window)
                v_rms_meas = sum(measured_window) / len(measured_window)

                # --- Calculation 3: Phase-A sampled maximum ---
                # At least two nominal cycles rather than the previous 5 ms.
                peak_window = vpeak_real_data[-peak_window_samples:]
                v_peak_real = max(abs(x) for x in peak_window)

                writer.writerow([
                    trial_id,
                    evaluation_stage,
                    val,
                    attack_type,
                    v_rms_real,
                    v_rms_meas,
                    v_peak_real,
                    tripped,
                    trip_time,
                    trip_time_status,
                    "valid",
                ])
                csvfile.flush()

                app.print_message(
                    f"Mag {val}: "
                    f"ProtectionHighInBuffer={tripped}, "
                    f"VRMS_Real={v_rms_real:.6f} kV, "
                    f"VPROT={v_rms_meas:.6f} pu, "
                    f"PhaseA_MaxAbs={v_peak_real:.6f} kV"
                )

            except Exception as exc:
                # Keep failed measurements visible in the results.
                writer.writerow([
                    trial_id,
                    evaluation_stage,
                    val,
                    attack_type,
                    "", "", "", "", "",
                    "unavailable",
                    f"invalid: {type(exc).__name__}: {exc}",
                ])
                csvfile.flush()
                app.print_message(f"Test at {val} kV failed: {exc}")

            finally:
                # Stop before continuing; explicitly restore zero bias.
                case.stop()
                atk_slider.value = 0.0
                time.sleep(1)
