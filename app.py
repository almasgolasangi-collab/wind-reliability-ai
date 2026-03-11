import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- 1. IEC 61400-1 DATA ENGINE ---
def generate_iec_wind(intensity="medium"):
    """Generates wind based on IEC 61400-1 Normal Turbulence Model (NTM)"""
    if intensity == "high": 
        v_hub = 22.5; i_ref = 0.16 # IEC Class A
    elif intensity == "low": 
        v_hub = 8.0; i_ref = 0.12  # IEC Class B
    else: 
        v_hub = 14.8; i_ref = 0.14 # Standard
    
    sigma = i_ref * (0.75 * v_hub + 5.6)
    # Ensure we don't get negative wind speeds
    data = np.random.normal(v_hub, sigma, 10)
    return np.clip(data, 0.1, 40)

# --- 2. MULTI-METHOD CALCULATION ENGINE ---
def calculate_metrics(avg_v, std_v):
    ti = std_v / avg_v 
    
    # Stress Index based on physical pressure
    stress_idx = np.clip(((avg_v**2)/(25**2)) + (ti * 1.5), 0, 1)
    stress_lbl = "CRITICAL" if stress_idx > 0.75 else "MODERATE" if stress_idx > 0.4 else "LOW"

    # AI PARAMETERS (Vague Logic)
    mid = stress_idx 
    pi = np.clip(ti * 2.2, 0.1, 0.45) # Hesitation degree
    L, U = np.clip(mid - 0.12, 0, 1), np.clip(mid + 0.12, 0, 1)
    
    # Vague Truth (t) and Falsehood (f)
    t = mid * (1 - pi)
    f = (1 - mid) * (1 - pi)
    
    # Fault Tree (Hard Limit at 20m/s)
    ft_risk = 1.0 if avg_v >= 20.0 else 0.0
    
    # Markov Chain
    markov_risk = np.clip(avg_v / 26.0, 0, 1)

    return (ti, stress_idx, stress_lbl, L, mid, U, t, f, pi), ft_risk, markov_risk

# --- 3. THE 3-BRAIN AUTONOMOUS DECISION LAYER ---
def get_autonomous_decisions(r_ft, r_mar, r_mc, stress_lbl):
    d_ft = ("SAFE (Normal)", "green") if r_ft > 0.5 else ("HALT (Over-limit)", "red")
    d_mar = ("OPTIMAL", "green") if r_mar > 0.75 else ("CAUTION", "orange") if r_mar > 0.4 else ("SHUTDOWN", "red")
    
    if r_mc > 0.82: d_mc = (f"SAFE: {stress_lbl}", "green")
    elif r_mc > 0.50: d_mc = (f"WARNING: {stress_lbl}", "orange")
    else: d_mc = (f"EMERGENCY: {stress_lbl}", "red")
        
    return d_ft, d_mar, d_mc

def main():
    print("\n" + "="*60 + "\n GEN-AI WIND RELIABILITY: MULTI-METHOD DASHBOARD \n" + "="*60)
    
    try:
        mode = input("Select Mode ('real' or 'gen'): ").lower()
        if mode == 'gen':
            intensity = input("Intensity (low, med, high): ").lower()
            data = generate_iec_wind(intensity)
        else:
            raw_input = input("Wind Speeds (comma separated): ")
            data = [float(x) for x in raw_input.split(",")]
        
        avg_v, std_v = np.mean(data), np.std(data, ddof=1)
        metrics, ft_r, mar_r = calculate_metrics(avg_v, std_v)
        (ti, stress_idx, stress_lbl, L, M, U, t, f, pi) = metrics
        
        rel_ft, rel_mar, rel_mc = 1 - ft_r, 1 - mar_r, 1 - M
        dec_ft, dec_mar, dec_mc = get_autonomous_decisions(rel_ft, rel_mar, rel_mc, stress_lbl)

        # --- PLOTTING ---
        fig = plt.figure(figsize=(15, 9), facecolor='#f0f2f5')
        gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1])
        ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

        # PANEL 1: FUSION MAP
        alphas = np.linspace(0, 1, 20)
        # Vague Area
        ax1.fill_betweenx(alphas, t*alphas, (1-f)+(f*(1-alphas)), color='gold', alpha=0.2, label='Vague Uncertainty')
        # Monte Carlo Area
        ax1.fill_betweenx(alphas, L+alphas*(M-L), U-alphas*(U-M), color='blue', alpha=0.15, label='MC Fuzzy Zone')
        
        ax1.axvline(ft_r, color='black', ls='--', lw=2, label='Fault Tree Risk')
        ax1.axvline(mar_r, color='darkorange', ls=':', lw=2, label='Markov Risk')
        ax1.axvline(M, color='blue', ls='-', lw=2, label='AI Risk Index')
        
        ax1.set_title(f"INTELLIGENCE FUSION MAP\nTurbulence (TI): {ti:.2f} | Stress: {stress_idx*100:.1f}%", fontweight='bold')
        ax1.set_xlim(-0.05, 1.05); ax1.legend(loc='upper right', fontsize='small')

        # PANEL 2: COMPARISON
        bars = ax2.bar(['Fault Tree', 'Markov Chain', 'Monte Carlo AI'], [rel_ft, rel_mar, rel_mc], 
                        color=['#34495e', '#e67e22', '#2ecc71'], edgecolor='black')
        ax2.set_title(f"RELIABILITY COMPARISON\nDetected: {stress_lbl} STRESS", fontweight='bold')
        ax2.set_ylim(0, 1.2)
        for b in bars:
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.2f}", ha='center', fontweight='bold')

        # PANEL 3: DECISION LAYER
        ax3.set_axis_off()
        box = dict(boxstyle='round,pad=1', facecolor='white', edgecolor='#bdc3c7', linewidth=1)
        ax3.text(0.15, 0.4, f"FAULT TREE\n(BINARY):\n\n{dec_ft[0]}", color=dec_ft[1], fontsize=10, fontweight='bold', ha='center', bbox=box)
        ax3.text(0.50, 0.4, f"MARKOV CHAIN\n(PROBABILISTIC):\n\n{dec_mar[0]}", color=dec_mar[1], fontsize=10, fontweight='bold', ha='center', bbox=box)
        ax3.text(0.85, 0.4, f"MONTE CARLO AI\n(GENERATIVE):\n\n{dec_mc[0]}", color=dec_mc[1], fontsize=10, fontweight='bold', ha='center', 
                 bbox=dict(box, edgecolor=dec_mc[1], linewidth=3))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error occurred: {e}. Please ensure inputs are correct.")

if __name__ == "__main__":
    main()
