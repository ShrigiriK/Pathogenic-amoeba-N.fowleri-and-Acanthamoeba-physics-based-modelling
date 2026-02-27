"""
Mathematical Model for Acanthamoeba Growth in Kerala Water Bodies
Extended from N. fowleri framework for different amoeba species

Author: Dr. Jijo P Ulahannan
Institution: Government College Kasaragod  
Date: December 2025

Acanthamoeba differs from N. fowleri in several key aspects:
- Wider temperature tolerance (4-56°C vs 10-46°C for N. fowleri)
- More resistant cysts that survive harsh conditions longer
- Causes eye infections (keratitis) rather than brain infection
- More common in environment but less virulent
- Different optimal growth conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AcanthamoebaKeralaModel:
    """
    Mathematical model for Acanthamoeba population dynamics in Kerala water bodies.
    
    Key differences from N. fowleri:
    - Broader temperature tolerance
    - More persistent cysts
    - Different clinical implications (eye infections vs brain)
    - More ubiquitous in environment
    """
    
    def __init__(self, depth=3.0, kerala_climate=True):
        """
        Initialize Acanthamoeba model for Kerala water bodies.
        
        Parameters:
        -----------
        depth : float
            Water body depth in meters
        kerala_climate : bool
            Use Kerala-specific environmental parameters
        """
        # Physical parameters
        self.depth = depth
        self.nz = 30  # Depth grid points
        self.dz = depth / self.nz
        self.z = np.linspace(0, depth, self.nz)
        
        # Biological parameters - Acanthamoeba specific
        self.r_max = 0.3      # Maximum growth rate (slower than N. fowleri)
        self.K = 5e5          # Lower carrying capacity
        self.T_opt = 30.0     # Lower optimal temperature (30°C vs 40°C)
        self.T_min = 4.0      # Much lower minimum (can survive near freezing)
        self.T_max = 56.0     # Higher maximum temperature tolerance
        self.sigma = 12.0     # Broader temperature tolerance
        
        # Diffusion parameters (similar mobility)
        self.D_T = 0.0008     # Trophozoite diffusion (slightly slower)
        self.D_F = 0.0015     # Flagellate diffusion
        self.D_C = 0.00005    # Cyst diffusion (very slow)
        
        # Stage transition rates - Acanthamoeba specific
        self.k_TF = 0.05      # Lower T→F rate (less motile stage switching)
        self.k_FT = 0.15      # F→T reversion
        self.k_TC = 0.08      # Higher T→C (more prone to encystment)
        self.k_CT = 0.01      # Lower C→T (cysts more persistent)
        
        # Environmental tolerance parameters
        self.pH_opt = 7.2     # Optimal pH
        self.pH_min = 5.0     # Minimum pH
        self.pH_max = 9.0     # Maximum pH
        self.salinity_tolerance = 0.1  # Can handle some salinity
        
        # Kerala climate parameters
        if kerala_climate:
            self.T_surface_avg = 28.0   # Kerala average
            self.T_seasonal_amp = 3.0   # Seasonal variation
            self.T_daily_amp = 4.0      # Daily variation
            self.peak_day = 120         # April-May peak
            self.thermocline = 1.5      # Thermocline depth
            
        # Initialize concentration fields
        self.T = np.zeros(self.nz)  # Trophozoites
        self.F = np.zeros(self.nz)  # Flagellates
        self.C = np.zeros(self.nz)  # Cysts
        self.temperature = np.zeros(self.nz)
        self.pH = np.ones(self.nz) * 7.2  # Assume neutral pH initially
    
    def thermal_stratification(self, t, day_of_year):
        """
        Calculate temperature profile - same as N. fowleri but with different responses.
        """
        # Surface temperature cycles
        T_seasonal = self.T_seasonal_amp * np.sin(2*np.pi*(day_of_year - self.peak_day)/365)
        T_daily = self.T_daily_amp * np.sin(2*np.pi*(t%24 - 12)/24)
        T_surface = self.T_surface_avg + T_seasonal + T_daily
        
        # Deep water temperature
        T_deep = 25.0
        
        # Exponential decay with depth
        for i, z in enumerate(self.z):
            self.temperature[i] = T_surface * np.exp(-z/self.thermocline) + T_deep
    
    def temperature_factor(self, T):
        """
        Acanthamoeba temperature response - broader tolerance than N. fowleri.
        """
        # Broader Gaussian tolerance
        factor = np.exp(-((T - self.T_opt)**2) / (2 * self.sigma**2))
        
        # Apply survival limits (wider range)
        factor = factor * (T >= self.T_min) * (T <= self.T_max)
        
        return factor
    
    def pH_factor(self, pH):
        """
        pH dependence for Acanthamoeba growth.
        """
        # Optimal pH range
        if isinstance(pH, np.ndarray):
            factor = np.ones_like(pH)
            # Triangular response centered at pH_opt
            factor = np.where((pH >= self.pH_min) & (pH <= self.pH_max),
                            1 - np.abs(pH - self.pH_opt) / (self.pH_max - self.pH_min),
                            0)
        else:
            if self.pH_min <= pH <= self.pH_max:
                factor = 1 - abs(pH - self.pH_opt) / (self.pH_max - self.pH_min)
            else:
                factor = 0
        
        return np.maximum(0, factor)
    
    def environmental_stress(self, depth_idx):
        """
        Calculate environmental stress factors affecting growth and transitions.
        """
        temp_stress = 1 - self.temperature_factor(self.temperature[depth_idx])
        pH_stress = 1 - self.pH_factor(self.pH[depth_idx])
        
        # Nutrient availability (decreases with depth)
        nutrient_availability = np.maximum(0.1, 1 - self.z[depth_idx]/(2*self.depth))
        nutrient_stress = 1 - nutrient_availability
        
        # Combined stress (0 = no stress, 1 = maximum stress)
        total_stress = np.minimum(1.0, temp_stress + pH_stress + nutrient_stress)
        
        return total_stress, nutrient_availability
    
    def growth_rate(self, depth_idx):
        """
        Calculate effective growth rate considering all environmental factors.
        """
        temp_factor = self.temperature_factor(self.temperature[depth_idx])
        pH_factor = self.pH_factor(self.pH[depth_idx])
        _, nutrient_factor = self.environmental_stress(depth_idx)
        
        # Multiplicative effects
        effective_rate = self.r_max * temp_factor * pH_factor * nutrient_factor
        
        return effective_rate
    
    def stage_transitions(self, depth_idx):
        """
        Calculate stage transition rates based on environmental stress.
        
        Acanthamoeba encysts more readily under stress than N. fowleri.
        """
        stress, nutrient = self.environmental_stress(depth_idx)
        temp_factor = self.temperature_factor(self.temperature[depth_idx])
        
        # Modified transition rates based on environmental conditions
        k_TF = self.k_TF * (1 - nutrient) * temp_factor  # Low nutrients → flagellate
        k_FT = self.k_FT * temp_factor
        k_TC = self.k_TC * (1 + 3*stress**2)  # Stress → cyst (more sensitive than N. fowleri)
        k_CT = self.k_CT * temp_factor * nutrient * (1 - stress)  # Conservative excystation
        
        return k_TF, k_FT, k_TC, k_CT
    
    def diffusion_step(self, concentration, D, dt):
        """
        Apply diffusion operator using finite differences.
        """
        C_new = concentration.copy()
        
        # Interior points (central difference)
        for i in range(1, self.nz-1):
            d2C_dz2 = (concentration[i+1] - 2*concentration[i] + concentration[i-1]) / self.dz**2
            C_new[i] += D * dt * d2C_dz2
        
        # Boundary conditions (no flux)
        C_new[0] = C_new[1]
        C_new[-1] = C_new[-2]
        
        return C_new
    
    def reaction_step(self, dt):
        """
        Apply biological reactions and stage transitions.
        """
        for i in range(self.nz):
            # Get local parameters
            r_eff = self.growth_rate(i)
            k_TF, k_FT, k_TC, k_CT = self.stage_transitions(i)
            
            # Current concentrations
            T_curr = self.T[i]
            F_curr = self.F[i]
            C_curr = self.C[i]
            N_total = T_curr + F_curr + C_curr
            
            # Logistic growth factor
            growth_limit = max(0, 1 - N_total/self.K)
            
            # Rate equations
            dT_dt = (r_eff * T_curr * growth_limit - 
                    k_TF * T_curr - k_TC * T_curr + 
                    k_FT * F_curr + k_CT * C_curr)
            
            dF_dt = k_TF * T_curr - k_FT * F_curr
            
            dC_dt = k_TC * T_curr - k_CT * C_curr
            
            # Update concentrations
            self.T[i] = max(0, T_curr + dt * dT_dt)
            self.F[i] = max(0, F_curr + dt * dF_dt)
            self.C[i] = max(0, C_curr + dt * dC_dt)
    
    def environmental_update(self, t):
        """
        Update environmental parameters (pH can vary with biological activity).
        """
        # Simple pH model: biological activity can lower pH slightly
        base_pH = 7.2
        biological_activity = np.sum(self.T + self.F) / (self.K * self.nz)
        pH_shift = -0.5 * biological_activity  # High activity lowers pH
        
        for i in range(self.nz):
            self.pH[i] = base_pH + pH_shift + 0.1 * np.sin(2*np.pi*t/168)  # Weekly variation
    
    def simulate_step(self, t, dt):
        """
        Perform one time step of the simulation.
        """
        # Update environmental conditions
        day_of_year = (t/24) % 365
        self.thermal_stratification(t, day_of_year)
        self.environmental_update(t)
        
        # Apply diffusion
        self.T = self.diffusion_step(self.T, self.D_T, dt)
        self.F = self.diffusion_step(self.F, self.D_F, dt)
        self.C = self.diffusion_step(self.C, self.D_C, dt)
        
        # Apply reactions
        self.reaction_step(dt)
    
    def run_simulation(self, t_final=720, dt=0.1, save_interval=2.0):
        """
        Run complete Acanthamoeba simulation.
        
        Parameters:
        -----------
        t_final : float
            Simulation time in hours (default: 720 = 30 days)
        dt : float
            Time step in hours
        save_interval : float
            Data saving interval in hours
        """
        n_steps = int(t_final / dt)
        save_every = int(save_interval / dt)
        
        # Storage arrays
        times = []
        T_profiles = []
        F_profiles = []
        C_profiles = []
        temp_profiles = []
        pH_profiles = []
        total_populations = []
        keratitis_risk = []  # Eye infection risk
        
        print(f"Running Acanthamoeba simulation for {t_final} hours")
        print(f"Temperature range: {self.T_min}-{self.T_max}°C (optimal: {self.T_opt}°C)")
        print(f"pH range: {self.pH_min}-{self.pH_max} (optimal: {self.pH_opt})")
        
        for step in range(n_steps):
            t = step * dt
            
            # Perform simulation step
            self.simulate_step(t, dt)
            
            # Save data
            if step % save_every == 0:
                times.append(t)
                T_profiles.append(self.T.copy())
                F_profiles.append(self.F.copy())
                C_profiles.append(self.C.copy())
                temp_profiles.append(self.temperature.copy())
                pH_profiles.append(self.pH.copy())
                
                total_pop = np.sum(self.T + self.F + self.C) * self.dz
                total_populations.append(total_pop)
                
                # Keratitis risk (surface concentration × contact probability)
                surface_risk = self.T[0] * 0.01  # 1% contact probability
                keratitis_risk.append(surface_risk)
                
                if step % (save_every * 50) == 0:
                    print(f"  Time: {t:.1f} hrs, Total: {total_pop:.2e}, Surface T: {self.temperature[0]:.1f}°C")
        
        results = {
            'times': np.array(times),
            'T_profiles': np.array(T_profiles),
            'F_profiles': np.array(F_profiles),
            'C_profiles': np.array(C_profiles),
            'temp_profiles': np.array(temp_profiles),
            'pH_profiles': np.array(pH_profiles),
            'total_populations': np.array(total_populations),
            'keratitis_risk': np.array(keratitis_risk),
            'depths': self.z
        }
        
        return results
    
    def calculate_keratitis_risk(self, surface_concentration):
        """
        Calculate keratitis (eye infection) risk from surface concentration.
        
        Risk factors:
        - Contact lens use
        - Swimming/water sports
        - Eye trauma with contaminated water
        """
        # Risk threshold much lower than N. fowleri (eye vs brain)
        base_risk_threshold = 100  # cells/mL
        
        # Risk categories
        if surface_concentration < base_risk_threshold:
            risk_level = "Low"
            risk_factor = surface_concentration / base_risk_threshold
        elif surface_concentration < 5 * base_risk_threshold:
            risk_level = "Moderate"
            risk_factor = 1 + (surface_concentration - base_risk_threshold) / base_risk_threshold
        else:
            risk_level = "High" 
            risk_factor = min(10, surface_concentration / base_risk_threshold)
        
        return risk_level, risk_factor


def plot_acanthamoeba_results(results, model):
    """
    Create comprehensive plots for Acanthamoeba simulation results.
    """
    fig = plt.figure(figsize=(18, 14))
    
    # Plot 1: Population dynamics
    plt.subplot(3, 4, 1)
    plt.semilogy(results['times']/24, results['total_populations'], 'b-', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Total Population (cells)')
    plt.title('Acanthamoeba Population Growth')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Temperature and pH effects
    plt.subplot(3, 4, 2)
    surface_temps = results['temp_profiles'][:, 0]
    surface_pH = results['pH_profiles'][:, 0]
    
    ax2a = plt.gca()
    line1 = ax2a.plot(results['times']/24, surface_temps, 'r-', linewidth=2, label='Temperature')
    ax2a.set_ylabel('Temperature (°C)', color='r')
    ax2a.axhline(model.T_opt, color='r', linestyle='--', alpha=0.5)
    
    ax2b = ax2a.twinx()
    line2 = ax2b.plot(results['times']/24, surface_pH, 'g-', linewidth=2, label='pH')
    ax2b.set_ylabel('pH', color='g')
    ax2b.axhline(model.pH_opt, color='g', linestyle='--', alpha=0.5)
    
    ax2a.set_xlabel('Time (days)')
    plt.title('Environmental Conditions')
    
    # Plot 3: Life stage distribution
    plt.subplot(3, 4, 3)
    total_T = np.sum(results['T_profiles'], axis=1) * model.dz
    total_F = np.sum(results['F_profiles'], axis=1) * model.dz
    total_C = np.sum(results['C_profiles'], axis=1) * model.dz
    
    plt.plot(results['times']/24, total_T, 'r-', linewidth=2, label='Trophozoite')
    plt.plot(results['times']/24, total_F, 'g-', linewidth=2, label='Flagellate') 
    plt.plot(results['times']/24, total_C, 'b-', linewidth=2, label='Cyst')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('Life Stage Dynamics')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Depth distribution (final)
    plt.subplot(3, 4, 4)
    final_T = results['T_profiles'][-1, :]
    final_F = results['F_profiles'][-1, :]
    final_C = results['C_profiles'][-1, :]
    
    plt.semilogx(final_T + 1, model.z, 'r-', linewidth=2, label='Trophozoite')
    plt.semilogx(final_F + 1, model.z, 'g-', linewidth=2, label='Flagellate')
    plt.semilogx(final_C + 1, model.z, 'b-', linewidth=2, label='Cyst')
    plt.ylabel('Depth (m)')
    plt.xlabel('Concentration (cells/mL)')
    plt.title('Final Depth Distribution')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Temperature tolerance comparison
    plt.subplot(3, 4, 5)
    T_range = np.linspace(0, 60, 200)
    acanthamoeba_factor = [model.temperature_factor(T) for T in T_range]
    
    # Compare with N. fowleri (for reference)
    nfowleri_T_opt = 40
    nfowleri_sigma = 8
    nfowleri_factor = [np.exp(-((T - nfowleri_T_opt)**2)/(2*nfowleri_sigma**2)) 
                      * (T >= 10) * (T <= 46) for T in T_range]
    
    plt.plot(T_range, acanthamoeba_factor, 'b-', linewidth=2, label='Acanthamoeba')
    plt.plot(T_range, nfowleri_factor, 'r--', linewidth=2, label='N. fowleri (ref)')
    plt.axvline(model.T_opt, color='b', linestyle=':', alpha=0.7)
    plt.axvline(nfowleri_T_opt, color='r', linestyle=':', alpha=0.7)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Growth Factor')
    plt.title('Temperature Tolerance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Spatiotemporal evolution
    plt.subplot(3, 4, 6)
    T_log = np.log10(results['T_profiles'] + 1)
    im = plt.imshow(T_log.T, aspect='auto', origin='upper', cmap='Reds',
                   extent=[results['times'][0]/24, results['times'][-1]/24, 
                          model.depth, 0])
    plt.colorbar(im, label='log₁₀(Trophozoites + 1)')
    plt.xlabel('Time (days)')
    plt.ylabel('Depth (m)')
    plt.title('Trophozoite Evolution')
    
    # Plot 7: Keratitis risk assessment
    plt.subplot(3, 4, 7)
    risk_levels = results['keratitis_risk']
    risk_threshold = 1.0  # Risk factor = 1
    
    colors = ['green' if r < 0.5 else 'orange' if r < 2.0 else 'red' for r in risk_levels]
    plt.scatter(results['times']/24, risk_levels, c=colors, alpha=0.6, s=20)
    plt.axhline(risk_threshold, color='red', linestyle='--', label='High Risk Threshold')
    plt.xlabel('Time (days)')
    plt.ylabel('Keratitis Risk Factor')
    plt.title('Eye Infection Risk')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: pH effects on growth
    plt.subplot(3, 4, 8)
    pH_range = np.linspace(4, 10, 100)
    pH_factors = [model.pH_factor(pH) for pH in pH_range]
    
    plt.plot(pH_range, pH_factors, 'g-', linewidth=2)
    plt.axvline(model.pH_opt, color='g', linestyle='--', alpha=0.7, label='Optimal pH')
    plt.axvspan(model.pH_min, model.pH_max, alpha=0.2, color='green', label='Survival range')
    plt.xlabel('pH')
    plt.ylabel('Growth Factor')
    plt.title('pH Dependence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Environmental stress factors
    plt.subplot(3, 4, 9)
    depths = model.z
    stress_factors = []
    nutrient_factors = []
    
    for i in range(len(depths)):
        stress, nutrient = model.environmental_stress(i)
        stress_factors.append(stress)
        nutrient_factors.append(nutrient)
    
    plt.plot(stress_factors, depths, 'r-', linewidth=2, label='Stress')
    plt.plot(nutrient_factors, depths, 'b-', linewidth=2, label='Nutrients')
    plt.xlabel('Factor (0-1)')
    plt.ylabel('Depth (m)')
    plt.title('Environmental Gradients')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Plot 10: Cyst persistence
    plt.subplot(3, 4, 10)
    cyst_surface = results['C_profiles'][:, 0]  # Surface cysts
    cyst_deep = results['C_profiles'][:, -5:]  # Deep cysts (bottom 5 layers)
    cyst_deep_mean = np.mean(cyst_deep, axis=1)
    
    plt.semilogy(results['times']/24, cyst_surface, 'b-', linewidth=2, label='Surface')
    plt.semilogy(results['times']/24, cyst_deep_mean, 'r-', linewidth=2, label='Deep')
    plt.xlabel('Time (days)')
    plt.ylabel('Cyst Concentration')
    plt.title('Cyst Persistence by Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 11: Comparison summary
    plt.subplot(3, 4, 11)
    comparison_data = [
        ['Parameter', 'Acanthamoeba', 'N. fowleri'],
        ['T_optimal', f'{model.T_opt}°C', '40°C'],
        ['T_range', f'{model.T_min}-{model.T_max}°C', '10-46°C'],
        ['Growth rate', f'{model.r_max} hr⁻¹', '0.5 hr⁻¹'],
        ['Infection', 'Keratitis (eye)', 'PAM (brain)'],
        ['Mortality', '~5%', '~97%'],
        ['Ubiquity', 'Very common', 'Uncommon']
    ]
    
    # Create text table
    table_text = ""
    for row in comparison_data:
        table_text += f"{row[0]:<12} {row[1]:<15} {row[2]:<15}\n"
    
    plt.text(0.05, 0.95, table_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('Species Comparison')
    
    # Plot 12: Summary statistics
    plt.subplot(3, 4, 12)
    stats_text = f"""
SIMULATION SUMMARY:
Duration: {results['times'][-1]/24:.1f} days
Peak Population: {results['total_populations'].max():.2e}
Final Population: {results['total_populations'][-1]:.2e}
Max Keratitis Risk: {results['keratitis_risk'].max():.2f}

ENVIRONMENTAL CONDITIONS:
Temp Range: {results['temp_profiles'].min():.1f}-{results['temp_profiles'].max():.1f}°C
pH Range: {results['pH_profiles'].min():.2f}-{results['pH_profiles'].max():.2f}
Water Depth: {model.depth:.1f} m

RISK ASSESSMENT:
High Risk Days: {np.sum(results['keratitis_risk'] > 1.0):.0f}
Surface Peak: {results['T_profiles'][:, 0].max():.1e} cells/mL
Cyst Reservoir: {results['C_profiles'][-1, :].sum():.1e} cells

KERALA RELEVANCE:
Suitable temp year-round
Contact lens users at risk
Swimming/bathing exposure
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.axis('off')
    plt.title('Model Summary')
    
    plt.tight_layout()
    plt.suptitle('Acanthamoeba Population Dynamics in Kerala Water Bodies', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def run_acanthamoeba_parameter_study():
    """
    Study parameter sensitivity for Acanthamoeba model.
    """
    print("Running Acanthamoeba parameter sensitivity study...")
    
    # Base case
    base_model = AcanthamoebaKeralaModel(depth=3.0)
    base_model.T[0:3] = 500   # Initial surface contamination
    base_model.C[5:10] = 200  # Cysts in sediment
    
    base_results = base_model.run_simulation(t_final=360, dt=0.1)  # 15 days
    
    # Parameter variations
    variations = {
        'temperature_optimum': [25, 27, 30, 33, 35],
        'pH_optimum': [6.5, 7.0, 7.2, 7.5, 8.0],
        'carrying_capacity': [1e5, 2e5, 5e5, 1e6, 2e6]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (param, values) in enumerate(variations.items()):
        final_populations = []
        risk_levels = []
        
        for value in values:
            test_model = AcanthamoebaKeralaModel(depth=3.0)
            
            if param == 'temperature_optimum':
                test_model.T_opt = value
            elif param == 'pH_optimum':
                test_model.pH_opt = value
            elif param == 'carrying_capacity':
                test_model.K = value
            
            # Set initial conditions
            test_model.T[0:3] = 500
            test_model.C[5:10] = 200
            
            # Run simulation
            test_results = test_model.run_simulation(t_final=360, dt=0.1)
            final_populations.append(test_results['total_populations'][-1])
            risk_levels.append(test_results['keratitis_risk'].max())
        
        # Plot results
        ax = axes[i]
        ax2 = ax.twinx()
        
        line1 = ax.plot(values, final_populations, 'bo-', linewidth=2, markersize=8, label='Population')
        line2 = ax2.plot(values, risk_levels, 'ro-', linewidth=2, markersize=8, label='Risk')
        
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Final Population', color='b')
        ax2.set_ylabel('Max Keratitis Risk', color='r')
        ax.set_title(f'Sensitivity to {param.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/acanthamoeba_sensitivity.png', dpi=300, bbox_inches='tight')
    return fig


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("Acanthamoeba Population Dynamics Model for Kerala Water Bodies")
    print("Government College Kasaragod - Dr. Jijo P Ulahannan")
    print("=" * 80)
    
    # Create Acanthamoeba model
    print("\n1. Setting up Acanthamoeba model for Kerala conditions...")
    model = AcanthamoebaKeralaModel(depth=3.0, kerala_climate=True)
    
    print(f"   Temperature tolerance: {model.T_min}-{model.T_max}°C (optimal: {model.T_opt}°C)")
    print(f"   pH tolerance: {model.pH_min}-{model.pH_max} (optimal: {model.pH_opt})")
    print(f"   Broader tolerance than N. fowleri, more persistent cysts")
    
    # Set initial conditions
    print("2. Setting initial contamination...")
    model.T[0:2] = 500    # Lower initial surface contamination  
    model.F[0] = 50       # Some flagellates
    model.C[3:10] = 150   # More persistent cysts in sediment
    
    # Run simulation
    print("3. Running simulation...")
    results = model.run_simulation(t_final=720, dt=0.1, save_interval=2.0)  # 30 days
    
    # Plot results
    print("4. Generating comprehensive plots...")
    fig = plot_acanthamoeba_results(results, model)
    plt.savefig('/mnt/user-data/outputs/acanthamoeba_kerala_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Parameter sensitivity
    print("5. Running parameter sensitivity study...")
    param_fig = run_acanthamoeba_parameter_study()
    plt.show()
    
    # Calculate final risk assessment
    print("\n6. Risk Assessment Summary:")
    surface_conc = results['T_profiles'][-1, 0]
    risk_level, risk_factor = model.calculate_keratitis_risk(surface_conc)
    
    print(f"   Final surface concentration: {surface_conc:.1e} cells/mL")
    print(f"   Keratitis risk level: {risk_level}")
    print(f"   Risk factor: {risk_factor:.2f}")
    print(f"   Peak population: {results['total_populations'].max():.2e} cells")
    print(f"   Days above risk threshold: {np.sum(results['keratitis_risk'] > 1.0)}")
    
    # Kerala-specific recommendations
    print("\n" + "=" * 80)
    print("KERALA-SPECIFIC RECOMMENDATIONS:")
    print("=" * 80)
    print("• Contact lens users: Use only sterile solutions, avoid water exposure")
    print("• Swimming/bathing: Avoid getting water in eyes, especially stagnant ponds")
    print("• Water storage: Regular cleaning of tanks, proper chlorination")
    print("• High-risk periods: Year-round due to favorable temperatures")
    print("• Monitoring: Surface water testing in public swimming areas")
    print("• Education: Awareness about eye protection during water activities")
    print("=" * 80)
    
    print("\nAcanthamoeba simulation complete!")
    print("Key differences from N. fowleri:")
    print("- Broader temperature tolerance (can grow at Kerala's year-round temps)")
    print("- More persistent cysts (environmental reservoir)")
    print("- Eye infections vs brain infections (lower mortality but more common)")
    print("- More ubiquitous in environment")
