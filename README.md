# Fourier-_Analysis-_-_FFT
"""
=====================================================================
  FOURIER ANALYSIS & SIGNAL DECOMPOSITION
  Author : Tanusree Saha
  Physics : B.Sc. Hons. Physics, Jogamaya Devi College
=====================================================================

WHAT THIS PROJECT DOES:
    Implements and applies Fourier analysis — the decomposition of
    any signal into sine and cosine waves (frequency components).

    f(x) = a₀/2 + Σ [aₙ cos(nωx) + bₙ sin(nωx)]

    METHODS:
        1. Fourier Series — periodic functions (analytical coefficients)
        2. Discrete Fourier Transform (DFT) — numerical computation
        3. Fast Fourier Transform (FFT) — O(N log N) algorithm
        4. Short-Time Fourier Transform (STFT) — time-frequency analysis

    PHYSICS APPLICATIONS:
        1. Square wave decomposition
        2. Gibbs phenomenon (overshoot at discontinuities)
        3. Quantum mechanics: momentum space from position space
        4. Diffraction pattern from crystal structure
        5. Heat equation solution via Fourier modes

    SIGNAL PROCESSING:
        1. Frequency filtering (low-pass, high-pass)
        2. Noise removal from a signal
        3. Financial time series spectral analysis

MATHEMATICAL CONCEPTS:
    - Fourier series convergence and Gibbs phenomenon
    - Parseval's theorem: energy conservation in frequency domain
    - Convolution theorem: multiplication in one domain = convolution
    - Nyquist-Shannon sampling theorem
    - Power spectral density

RELEVANCE TO FINANCE:
    FFT is used in options pricing (Carr-Madan FFT method),
    time series analysis of financial data (detecting cycles,
    seasonality), and signal processing of high-frequency
    trading data. Filtering out noise ≡ separating alpha from noise.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram, windows
from scipy.fft import fft, ifft, fftfreq, fftshift
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  1. FOURIER SERIES COEFFICIENTS
# ─────────────────────────────────────────────

def fourier_coefficients(f, T, n_terms=20, n_points=1000):
    """
    Compute Fourier series coefficients numerically.

    aₙ = (2/T) ∫₀ᵀ f(x) cos(2πnx/T) dx
    bₙ = (2/T) ∫₀ᵀ f(x) sin(2πnx/T) dx

    Then reconstruct:
    f(x) ≈ a₀/2 + Σ_{n=1}^{N} [aₙ cos(nωx) + bₙ sin(nωx)]
    """
    x  = np.linspace(0, T, n_points, endpoint=False)
    fx = f(x)
    dx = T / n_points
    omega = 2 * np.pi / T

    a = np.zeros(n_terms + 1)
    b = np.zeros(n_terms + 1)

    for n in range(n_terms + 1):
        a[n] = (2/T) * np.trapezoid(fx * np.cos(n * omega * x), x)
        b[n] = (2/T) * np.trapezoid(fx * np.sin(n * omega * x), x)

    return a, b, x, fx


def reconstruct_fourier(x, a, b, T, n_terms):
    """Reconstruct signal from Fourier coefficients."""
    omega = 2 * np.pi / T
    f_rec = a[0] / 2 * np.ones_like(x)
    for n in range(1, n_terms + 1):
        f_rec += a[n] * np.cos(n * omega * x) + b[n] * np.sin(n * omega * x)
    return f_rec


# ─────────────────────────────────────────────
#  2. STANDARD WAVEFORMS
# ─────────────────────────────────────────────

def square_wave(x, T=2*np.pi):
    """Square wave: +1 for first half, -1 for second half."""
    return np.where(x % T < T/2, 1.0, -1.0)


def sawtooth_wave(x, T=2*np.pi):
    """Sawtooth: linear ramp from -1 to +1 over each period."""
    return 2 * (x % T) / T - 1.0


def triangle_wave(x, T=2*np.pi):
    """Triangle wave: piecewise linear."""
    t = (x % T) / T
    return np.where(t < 0.5, 4*t - 1, 3 - 4*t)


# ─────────────────────────────────────────────
#  3. FFT & FREQUENCY DOMAIN
# ─────────────────────────────────────────────

def compute_fft(signal, dt):
    """
    Compute the FFT of a signal.

    The FFT decomposes a discrete signal into N complex
    exponentials e^(2πikt/N), giving amplitude and phase at
    each frequency.

    Time complexity: O(N log N) vs O(N²) for naive DFT.
    Invented by Cooley & Tukey (1965) — one of the most
    important algorithms in computational science.

    Parameters
    ----------
    signal : array – time-domain signal
    dt     : float – sampling interval

    Returns
    -------
    freqs      : array – frequency values (Hz)
    amplitudes : array – amplitude spectrum |X(f)|
    phases     : array – phase spectrum angle(X(f))
    """
    N         = len(signal)
    X         = fft(signal)
    freqs     = fftfreq(N, d=dt)
    amplitudes = np.abs(X) * 2 / N       # Two-sided → one-sided
    phases     = np.angle(X)

    # Positive frequencies only
    pos_mask   = freqs >= 0
    return freqs[pos_mask], amplitudes[pos_mask], phases[pos_mask], X


def low_pass_filter(signal, dt, cutoff_freq):
    """
    Low-pass filter: remove frequencies above cutoff.

    In frequency domain, filtering = multiply by a window function.
    Low-pass: keep low frequencies (smooth signal), zero out high freq.

    This is used in:
        - Audio processing (remove hiss)
        - Image processing (blur = low-pass)
        - Finance: smoothing price series to identify trends
    """
    freqs, _, _, X = compute_fft(signal, dt)
    N = len(signal)
    all_freqs = fftfreq(N, d=dt)

    X_filtered = X.copy()
    X_filtered[np.abs(all_freqs) > cutoff_freq] = 0.0

    return np.real(ifft(X_filtered))


def high_pass_filter(signal, dt, cutoff_freq):
    """
    High-pass filter: keep only high frequencies (edges/changes).
    Finance: isolates short-term fluctuations / noise.
    """
    freqs, _, _, X = compute_fft(signal, dt)
    N = len(signal)
    all_freqs = fftfreq(N, d=dt)

    X_filtered = X.copy()
    X_filtered[np.abs(all_freqs) < cutoff_freq] = 0.0

    return np.real(ifft(X_filtered))


# ─────────────────────────────────────────────
#  4. NOISY SIGNAL RECOVERY
# ─────────────────────────────────────────────

def noisy_signal_demo(t, signal_freqs=[1.0, 3.0, 7.0],
                      signal_amps=[1.0, 0.5, 0.3], noise_level=0.8):
    """
    Create a multi-frequency signal buried in noise.
    Then recover it using FFT filtering.

    This directly maps to separating 'signal' from 'noise'
    in financial time series — identifying real cycles vs random noise.
    """
    dt      = t[1] - t[0]
    # True signal
    signal  = sum(A * np.sin(2*np.pi*f*t)
                  for f, A in zip(signal_freqs, signal_amps))
    # Add Gaussian noise
    noise   = noise_level * np.random.normal(0, 1, len(t))
    noisy   = signal + noise

    # Recover via FFT low-pass filter
    cutoff = max(signal_freqs) * 1.5
    recovered = low_pass_filter(noisy, dt, cutoff)

    return signal, noisy, recovered


# ─────────────────────────────────────────────
#  5. PARSEVAL'S THEOREM VERIFICATION
# ─────────────────────────────────────────────

def parseval_check(signal, dt):
    """
    Parseval's theorem: total energy is conserved under FFT.

    Σ|x[n]|² = (1/N) Σ|X[k]|²

    Energy in time domain = Energy in frequency domain.
    This is the mathematical statement that Fourier transform
    is a unitary transformation — it preserves inner products.
    """
    N = len(signal)
    energy_time = np.sum(np.abs(signal)**2)
    X = fft(signal)
    energy_freq = np.sum(np.abs(X)**2) / N
    return energy_time, energy_freq


# ─────────────────────────────────────────────
#  6. VISUALISATION
# ─────────────────────────────────────────────

def plot_all():
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.58, wspace=0.38)

    BG     = '#0D1117'
    PANEL  = '#161B22'
    GRID_C = '#2A2A3A'
    TEXT_C = '#E0E0E0'
    GREEN  = '#00C896'
    RED    = '#FF4444'
    BLUE   = '#4A9EFF'
    GOLD   = '#FFD700'
    PURPLE = '#C084FC'
    AMBER  = '#FFB347'

    T  = 2 * np.pi
    x  = np.linspace(0, 2*T, 1000)

    # ── Panel 1: Square wave Fourier Decomposition ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(PANEL)
    f_sq = lambda x: square_wave(x, T)
    a_sq, b_sq, _, _ = fourier_coefficients(f_sq, T, n_terms=30)
    ax1.plot(x, f_sq(x), color='white', linewidth=1.5, alpha=0.6, label='Original')
    colors_n = [GREEN, BLUE, AMBER, RED, PURPLE]
    for i, n_terms in enumerate([1, 3, 7, 15, 29]):
        rec = reconstruct_fourier(x, a_sq, b_sq, T, n_terms)
        ax1.plot(x, rec, color=colors_n[i], linewidth=1.2,
                 label=f'N={n_terms} terms', alpha=0.85)
    ax1.set_title('Square Wave — Fourier Series Reconstruction\n'
                  '(Gibbs phenomenon: ~9% overshoot at discontinuities)',
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax1.set_xlabel('x', color=TEXT_C, fontsize=8)
    ax1.set_ylabel('f(x)', color=TEXT_C, fontsize=8)
    ax1.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT_C, ncol=3)
    ax1.tick_params(colors=TEXT_C, labelsize=7)
    ax1.grid(True, color=GRID_C, linewidth=0.4)
    for s in ax1.spines.values(): s.set_color(GRID_C)

    # ── Panel 2: Fourier Coefficients ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL)
    ns = np.arange(1, 20)
    bn_sq = b_sq[1:20]
    bn_theory = np.where(ns % 2 != 0, 4/(ns * np.pi), 0)
    ax2.bar(ns - 0.2, np.abs(bn_sq),     width=0.35, color=BLUE, alpha=0.8, label='Numerical |bₙ|')
    ax2.bar(ns + 0.2, np.abs(bn_theory), width=0.35, color=GOLD, alpha=0.6, label='Analytical |bₙ|')
    ax2.set_title('Fourier Coefficients bₙ\n(Square wave: non-zero only for odd n)',
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax2.set_xlabel('n', color=TEXT_C, fontsize=8)
    ax2.set_ylabel('|bₙ|', color=TEXT_C, fontsize=8)
    ax2.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT_C)
    ax2.tick_params(colors=TEXT_C, labelsize=7)
    ax2.grid(True, color=GRID_C, linewidth=0.4, axis='y')
    for s in ax2.spines.values(): s.set_color(GRID_C)

    # ── Panel 3: Multiple waveforms + spectra ──
    dt  = 0.001
    t_s = np.arange(0, 4, dt)
    waves = {
        'Square'  : (square_wave(2*np.pi*t_s), GREEN),
        'Sawtooth': (sawtooth_wave(2*np.pi*t_s), AMBER),
        'Triangle': (triangle_wave(2*np.pi*t_s), PURPLE),
    }

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor(PANEL)
    for i, (name, (sig, col)) in enumerate(waves.items()):
        ax3.plot(t_s[:500], sig[:500] + i*3, color=col,
                 linewidth=1.3, label=name, alpha=0.9)
    ax3.set_title('Waveforms: Square, Sawtooth, Triangle',
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax3.set_xlabel('Time', color=TEXT_C, fontsize=8)
    ax3.set_ylabel('Amplitude (offset)', color=TEXT_C, fontsize=8)
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    ax3.tick_params(colors=TEXT_C, labelsize=7)
    ax3.grid(True, color=GRID_C, linewidth=0.4)
    for s in ax3.spines.values(): s.set_color(GRID_C)

    # ── Panel 4: Frequency Spectra ──
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(PANEL)
    for name, (sig, col) in waves.items():
        freqs_s, amps, _, _ = compute_fft(sig, dt)
        mask = freqs_s <= 15
        ax4.plot(freqs_s[mask], amps[mask], color=col,
                 linewidth=1.3, label=name, alpha=0.9)
    ax4.set_title('Frequency Spectra (FFT)\n0–15 Hz',
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax4.set_xlabel('Frequency (Hz)', color=TEXT_C, fontsize=8)
    ax4.set_ylabel('Amplitude', color=TEXT_C, fontsize=8)
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    ax4.tick_params(colors=TEXT_C, labelsize=7)
    ax4.grid(True, color=GRID_C, linewidth=0.4)
    for s in ax4.spines.values(): s.set_color(GRID_C)

    # ── Panel 5: Noisy Signal Recovery ──
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor(PANEL)
    t_n  = np.linspace(0, 4, 2000)
    sig_true, sig_noisy, sig_rec = noisy_signal_demo(t_n)
    ax5.plot(t_n, sig_noisy, color=BLUE,  linewidth=0.6, alpha=0.5, label='Noisy signal')
    ax5.plot(t_n, sig_true,  color='#DDDDDD', linewidth=1.5,
             linestyle='--', alpha=0.7, label='True signal')
    ax5.plot(t_n, sig_rec,   color=GREEN, linewidth=1.8, label='Recovered (FFT filter)')
    ax5.set_title('Signal Recovery from Noise via FFT Low-Pass Filter\n'
                  '(SNR improvement: isolating 1Hz + 3Hz + 7Hz components)',
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax5.set_xlabel('Time', color=TEXT_C, fontsize=8)
    ax5.set_ylabel('Amplitude', color=TEXT_C, fontsize=8)
    ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    ax5.tick_params(colors=TEXT_C, labelsize=7)
    ax5.grid(True, color=GRID_C, linewidth=0.4)
    for s in ax5.spines.values(): s.set_color(GRID_C)

    # ── Panel 6: Spectrum of noisy vs clean ──
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL)
    dt_n = t_n[1] - t_n[0]
    f_noisy, a_noisy, _, _ = compute_fft(sig_noisy, dt_n)
    f_clean, a_clean, _, _ = compute_fft(sig_true,  dt_n)
    mask = f_noisy <= 20
    ax6.plot(f_noisy[mask], a_noisy[mask], color=BLUE,  linewidth=1,
             alpha=0.7, label='Noisy spectrum')
    ax6.plot(f_clean[mask], a_clean[mask], color=GREEN, linewidth=1.5,
             label='True signal spectrum')
    for f0 in [1.0, 3.0, 7.0]:
        ax6.axvline(f0, color=GOLD, linewidth=0.8, linestyle=':', alpha=0.6)
    ax6.set_title('Frequency Spectrum: Noisy vs True',
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax6.set_xlabel('Frequency (Hz)', color=TEXT_C, fontsize=8)
    ax6.set_ylabel('Amplitude', color=TEXT_C, fontsize=8)
    ax6.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_C)
    ax6.tick_params(colors=TEXT_C, labelsize=7)
    ax6.grid(True, color=GRID_C, linewidth=0.4)
    for s in ax6.spines.values(): s.set_color(GRID_C)

    # ── Panel 7: Chirp + STFT Spectrogram ──
    ax7 = fig.add_subplot(gs[3, :2])
    ax7.set_facecolor(PANEL)
    t_c   = np.linspace(0, 4, 8000)
    chirp = np.sin(2 * np.pi * (2 + 8 * t_c) * t_c)
    f_c, t_spec, Sxx = spectrogram(chirp, fs=2000, nperseg=256,
                                    noverlap=200, nfft=512)
    ax7.pcolormesh(t_spec, f_c[:50], 10*np.log10(Sxx[:50]+1e-10),
                   cmap='plasma', shading='gouraud')
    ax7.set_title('STFT Spectrogram — Chirp Signal (Frequency sweeping 2→40 Hz)\n'
                  'Short-Time Fourier Transform: time-frequency localisation',
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax7.set_xlabel('Time (s)', color=TEXT_C, fontsize=8)
    ax7.set_ylabel('Frequency (Hz)', color=TEXT_C, fontsize=8)
    ax7.tick_params(colors=TEXT_C, labelsize=7)
    for s in ax7.spines.values(): s.set_color(GRID_C)

    # ── Panel 8: Parseval's Theorem ──
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.set_facecolor(PANEL)
    test_signals = {
        'Square' : square_wave(2*np.pi*t_s),
        'Saw'    : sawtooth_wave(2*np.pi*t_s),
        'Triangle': triangle_wave(2*np.pi*t_s),
        'Chirp'  : chirp[:len(t_s)],
        'Noise'  : np.random.normal(0, 1, len(t_s)),
    }
    names, e_time, e_freq = [], [], []
    for name, sig in test_signals.items():
        et, ef = parseval_check(sig, dt)
        names.append(name)
        e_time.append(et)
        e_freq.append(ef)
    x_pos = np.arange(len(names))
    ax8.bar(x_pos - 0.2, e_time, width=0.35, color=BLUE,  alpha=0.8, label='Time domain')
    ax8.bar(x_pos + 0.2, e_freq, width=0.35, color=GREEN, alpha=0.8, label='Freq domain')
    ax8.set_title("Parseval's Theorem\nEnergy time = Energy frequency",
                  color=TEXT_C, fontsize=9, fontweight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(names, color=TEXT_C, fontsize=7)
    ax8.set_ylabel('Signal Energy', color=TEXT_C, fontsize=8)
    ax8.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT_C)
    ax8.tick_params(colors=TEXT_C, labelsize=7)
    ax8.grid(True, color=GRID_C, linewidth=0.4, axis='y')
    for s in ax8.spines.values(): s.set_color(GRID_C)

    fig.suptitle("Fourier Analysis & Signal Decomposition — From Wave Physics to Signal Processing",
                 color=TEXT_C, fontsize=13, fontweight='bold', y=1.01)

    plt.savefig('/mnt/user-data/outputs/fourier_analysis.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print("Chart saved.")


# ─────────────────────────────────────────────
#  7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 65)
    print("  FOURIER ANALYSIS & SIGNAL DECOMPOSITION")
    print("=" * 65)

    T  = 2 * np.pi
    x  = np.linspace(0, 2*T, 1000, endpoint=False)
    dt = 0.001

    # ── Fourier Coefficients ──
    print("\n── Fourier Coefficients: Square Wave bₙ ──")
    print("  (Analytical: bₙ = 4/nπ for odd n, 0 for even n)")
    f_sq = lambda x: square_wave(x, T)
    a_sq, b_sq, _, _ = fourier_coefficients(f_sq, T, n_terms=10)
    print(f"  {'n':>4}  {'bₙ (numerical)':>16}  {'4/nπ (exact)':>14}  {'error':>10}")
    for n in range(1, 11):
        exact = 4/(n*np.pi) if n % 2 != 0 else 0.0
        print(f"  {n:>4}  {b_sq[n]:>16.8f}  {exact:>14.8f}  "
              f"{abs(b_sq[n]-exact):>10.2e}")

    # ── Parseval's Theorem ──
    print("\n── Parseval's Theorem Verification ──")
    t_s = np.arange(0, 4, dt)
    for name, sig in [('Square',   square_wave(2*np.pi*t_s)),
                       ('Sawtooth', sawtooth_wave(2*np.pi*t_s)),
                       ('Triangle', triangle_wave(2*np.pi*t_s))]:
        et, ef = parseval_check(sig, dt)
        print(f"  {name:10s}: E_time = {et:.2f}  |  E_freq = {ef:.2f}  "
              f"|  ratio = {et/ef:.6f}")

    # ── Noise filtering ──
    print("\n── Signal Recovery from Noise ──")
    t_n = np.linspace(0, 4, 2000)
    sig_true, sig_noisy, sig_rec = noisy_signal_demo(t_n)
    snr_before = 10*np.log10(np.var(sig_true) / np.var(sig_noisy - sig_true))
    snr_after  = 10*np.log10(np.var(sig_true) / np.var(sig_rec  - sig_true))
    print(f"  SNR before filtering: {snr_before:.2f} dB")
    print(f"  SNR after filtering : {snr_after:.2f} dB")
    print(f"  Improvement         : {snr_after - snr_before:.2f} dB")

    # ── FFT accuracy ──
    print("\n── FFT Accuracy: Known Frequencies ──")
    t_test  = np.arange(0, 2, dt)
    for f0 in [1.0, 5.0, 10.0, 50.0]:
        sig = np.sin(2*np.pi*f0*t_test)
        freqs, amps, _, _ = compute_fft(sig, dt)
        peak_freq = freqs[np.argmax(amps[1:])+1]
        print(f"  True freq: {f0:>6.1f} Hz  |  "
              f"FFT peak: {peak_freq:>6.2f} Hz  |  "
              f"Error: {abs(peak_freq-f0):.4f} Hz")

    print("\n── Generating Charts ──")
    plot_all()
    print("\nDone.")
