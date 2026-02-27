"""
Simplified N. fowleri Growth Model for Tropical Water Bodies
Student Implementation for B.Sc. Physics Project

Author: Dr. Jijo Ulahannan (extending work by Shrigiri K)
Date: December 19, 2025
Purpose: Educational implementation for undergraduate physics students
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class SimpleNFowleriModel:
    """
    Simplified model for N. fowleri growth in tropical water body.
    
    This version focuses on the key physics concepts while remaining 
    accessible to B.Sc. Physics students.
    """
    
    def __init__(self, depth=3.0, kerala_climate=True):
        """
        Initialize model for a typical Kerala pond.
        
        Parameters:
        -----------
        depth : float
            Water body depth in meters
        kerala_climate : bool
            Use Kerala-specific temperature parameters
        """
        # Physical parameters
        self.depth = depth
        self.nz = 30  # Depth grid points
        self.dz = depth / self.nz  # Depth spacing
        self.z = np.linspace(0, depth, self.nz)  # Depth coordinates
        
        # Biological parameters (based on literature)
        self.r_max = 0.5      # Maximum growth rate (per hour)
        self.K = 1e6          # Carrying capacity (cells/mL)
        self.T_opt = 40.0     # Optimal temperature (°C)
        self.T_min = 10.0     # Minimum survival temperature
        self.T_max = 46.0     # Maximum survival temperature
        self.sigma = 8.0      # Temperature tolerance (°C)
        
        # Diffusion parameters
        self.D_T = 0.001      # Trophozoite diffusion (m²/hr)
        self.D_F = 0.002      # Flagellate diffusion (m²/hr)  
        self.D_C = 0.0001     # Cyst diffusion (m²/hr)
        
        # Stage transition rates (per hour)
        self.k_TF = 0.1       # Trophozoite → Flagellate
        self.k_FT = 0.2       # Flagellate → Trophozoite
        self.k_TC = 0.05      # Trophozoite → Cyst (stress)
        self.k_CT = 0.02      # Cyst → Trophozoite
        
        # Kerala climate parameters
        if kerala_climate:
            self.T_surface_avg = 28.0   # Average surface temperature
            self.T_seasonal_amp = 3.0   # Seasonal amplitude
            self.T_daily_amp = 4.0      # Daily amplitude
            self.peak_day = 120         # Peak temperature day (April-May)
            self.thermocline = 1.5      # Thermocline depth (m)
        
        # Initialize concentration fields [depth]
        self.T = np.zeros(self.nz)  # Trophozoites
        self.F = np.zeros(self.nz)  # Flagellates
        self.C = np.zeros(self.nz)  # Cysts
        self.temperature = np.zeros(self.nz)
    
    def thermal_stratification(self, t, day_of_year):
        """
        Calculate depth-dependent temperature profile.
        
        Models typical tropical pond thermal stratification with:
        - Seasonal variation (annual cycle)
        - Daily variation (diurnal cycle) 
        - Exponential decay with depth
        
        Parameters:
        -----------
        t : float
            Time in hours
        day_of_year : float
            Day number (1-365)
        """
        # Surface temperature with seasonal and daily cycles
        T_seasonal = self.T_seasonal_amp * np.sin(2*np.pi*(day_of_year - self.peak_day)/365)
        T_daily = self.T_daily_amp * np.sin(2*np.pi*(t%24 - 12)/24)
        T_surface = self.T_surface_avg + T_seasonal + T_daily
        
        # Deep water temperature (relatively constant)
        T_deep = 25.0
        
        # Exponential decay with depth (thermocline effect)
        for i, z in enumerate(self.z):
            self.temperature[i] = T_surface * np.exp(-z/self.thermocline) + T_deep
    
    def temperature_factor(self, T):
        """
        Calculate growth rate factor based on temperature.
        
        Uses Gaussian function centered at optimal temperature with
        cutoffs at minimum and maximum survival temperatures.
        """
        # Thermal tolerance (Gaussian-like)
        factor = np.exp(-((T - self.T_opt)**2) / (2 * self.sigma**2))
        
        # Apply survival limits
        factor = factor * (T >= self.T_min) * (T <= self.T_max)
        
        return factor
    
    def growth_rate(self, depth_idx):
        """Calculate effective growth rate at given depth."""
        temp_factor = self.temperature_factor(self.temperature[depth_idx])
        return self.r_max * temp_factor
    
    def stage_transitions(self, depth_idx):
        """
        Calculate stage transition rates based on environmental conditions.
        
        Stress factors:
        - Temperature stress (far from optimal)
        - Nutrient stress (simplified as depth-dependent)
        """
        temp_factor = self.temperature_factor(self.temperature[depth_idx])
        stress = max(0, 1 - temp_factor)  # Higher stress when temp suboptimal
        nutrient = max(0.1, 1 - self.z[depth_idx]/self.depth)  # Higher nutrients near surface
        
        # Modified transition rates
        k_TF = self.k_TF * (1 - nutrient)  # Low nutrients → flagellate
        k_FT = self.k_FT * temp_factor
        k_TC = self.k_TC * (1 + 5*stress**2)  # Stress → cyst
        k_CT = self.k_CT * temp_factor * nutrient  # Good conditions → excyst
        
        return k_TF, k_FT, k_TC, k_CT
    
    def diffusion_step(self, concentration, D, dt):
        """
        Apply diffusion operator using finite differences.
        
        Solves: ∂C/∂t = D ∂²C/∂z²
        """
        C_new = concentration.copy()
        
        # Interior points (central difference)
        for i in range(1, self.nz-1):
            d2C_dz2 = (concentration[i+1] - 2*concentration[i] + concentration[i-1]) / self.dz**2
            C_new[i] += D * dt * d2C_dz2
        
        # Boundary conditions (no flux at top and bottom)
        C_new[0] = C_new[1]
        C_new[-1] = C_new[-2]
        
        return C_new
    
    def reaction_step(self, dt):
        """
        Apply biological reactions and stage transitions.
        
        This is the core biological model incorporating:
        - Logistic growth of trophozoites
        - Stage transitions based on environmental conditions
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
            
            # Rate equations (from the mathematical model)
            dT_dt = (r_eff * T_curr * growth_limit - 
                    k_TF * T_curr - k_TC * T_curr + 
                    k_FT * F_curr + k_CT * C_curr)
            
            dF_dt = k_TF * T_curr - k_FT * F_curr
            
            dC_dt = k_TC * T_curr - k_CT * C_curr
            
            # Update concentrations
            self.T[i] = max(0, T_curr + dt * dT_dt)
            self.F[i] = max(0, F_curr + dt * dF_dt)
            self.C[i] = max(0, C_curr + dt * dC_dt)
    
    def simulate_step(self, t, dt):
        """Perform one time step of the simulation."""
        # Update temperature profile
        day_of_year = (t/24) % 365
        self.thermal_stratification(t, day_of_year)
        
        # Apply diffusion
        self.T = self.diffusion_step(self.T, self.D_T, dt)
        self.F = self.diffusion_step(self.F, self.D_F, dt)
        self.C = self.diffusion_step(self.C, self.D_C, dt)
        
        # Apply reactions
        self.reaction_step(dt)
    
    def run_simulation(self, t_final=720, dt=0.1, save_interval=1.0):
        """
        Run complete simulation.
        
        Parameters:
        -----------
        t_final : float
            Simulation time in hours (default: 720 = 30 days)
        dt : float
            Time step in hours (default: 0.1)
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
        total_populations = []
        
        print(f"Running simulation for {t_final} hours with dt = {dt}")
        print(f"Total steps: {n_steps}, saving every {save_every} steps")
        
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
                
                total_pop = np.sum(self.T + self.F + self.C) * self.dz  # Integrate over depth
                total_populations.append(total_pop)
                
                if step % (save_every * 100) == 0:
                    print(f"  Time: {t:.1f} hrs ({t/24:.1f} days), Total: {total_pop:.2e}")
        
        # Convert to numpy arrays for easier manipulation
        results = {
            'times': np.array(times),
            'T_profiles': np.array(T_profiles),
            'F_profiles': np.array(F_profiles), 
            'C_profiles': np.array(C_profiles),
            'temp_profiles': np.array(temp_profiles),
            'total_populations': np.array(total_populations),
            'depths': self.z
        }
        
        return results


def plot_simulation_results(results, model):
    """Create comprehensive plots of simulation results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Total population over time
    plt.subplot(3, 3, 1)
    plt.semilogy(results['times']/24, results['total_populations'], 'b-', linewidth=2)
    plt.xlabel('Time (days)')
    plt.ylabel('Total Population (cells)')
    plt.title('Population Growth Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Temperature variation over time (surface)
    plt.subplot(3, 3, 2)
    surface_temps = results['temp_profiles'][:, 0]
    plt.plot(results['times']/24, surface_temps, 'r-', linewidth=2)
    plt.axhline(model.T_opt, color='g', linestyle='--', label=f'Optimal ({model.T_opt}°C)')
    plt.xlabel('Time (days)')
    plt.ylabel('Surface Temperature (°C)')
    plt.title('Surface Temperature Variation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Final depth profile of organisms
    plt.subplot(3, 3, 3)
    final_T = results['T_profiles'][-1, :]
    final_F = results['F_profiles'][-1, :]
    final_C = results['C_profiles'][-1, :]
    
    plt.plot(np.log10(final_T + 1), model.z, 'r-', linewidth=2, label='Trophozoite')
    plt.plot(np.log10(final_F + 1), model.z, 'g-', linewidth=2, label='Flagellate')
    plt.plot(np.log10(final_C + 1), model.z, 'b-', linewidth=2, label='Cyst')
    plt.ylabel('Depth (m)')
    plt.xlabel('log₁₀(Concentration + 1)')
    plt.title('Final Depth Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    
    # Plot 4: Temperature depth profile
    plt.subplot(3, 3, 4)
    final_temp = results['temp_profiles'][-1, :]
    plt.plot(final_temp, model.z, 'r-', linewidth=2)
    plt.axvline(model.T_opt, color='g', linestyle='--', alpha=0.5, label='Optimal')
    plt.axvspan(model.T_min, model.T_max, alpha=0.2, color='orange', label='Survival range')
    plt.ylabel('Depth (m)')
    plt.xlabel('Temperature (°C)')
    plt.title('Temperature vs Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    
    # Plot 5: Spatiotemporal evolution (Trophozoites)
    plt.subplot(3, 3, 5)
    T_log = np.log10(results['T_profiles'] + 1)
    im = plt.imshow(T_log.T, aspect='auto', origin='upper', cmap='Reds',
                   extent=[results['times'][0]/24, results['times'][-1]/24, 
                          model.depth, 0])
    plt.colorbar(im, label='log₁₀(Trophozoites + 1)')
    plt.xlabel('Time (days)')
    plt.ylabel('Depth (m)')
    plt.title('Trophozoite Evolution')
    
    # Plot 6: Growth rate factor vs temperature
    plt.subplot(3, 3, 6)
    T_range = np.linspace(0, 50, 100)
    growth_factors = [model.temperature_factor(T) for T in T_range]
    plt.plot(T_range, growth_factors, 'b-', linewidth=2)
    plt.axvline(model.T_opt, color='g', linestyle='--', label='Optimal')
    plt.axvspan(model.T_min, model.T_max, alpha=0.2, color='orange', label='Survival')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Growth Factor')
    plt.title('Temperature Dependence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Stage composition over time
    plt.subplot(3, 3, 7)
    total_T = np.sum(results['T_profiles'], axis=1) * model.dz
    total_F = np.sum(results['F_profiles'], axis=1) * model.dz
    total_C = np.sum(results['C_profiles'], axis=1) * model.dz
    
    plt.plot(results['times']/24, total_T, 'r-', linewidth=2, label='Trophozoite')
    plt.plot(results['times']/24, total_F, 'g-', linewidth=2, label='Flagellate')
    plt.plot(results['times']/24, total_C, 'b-', linewidth=2, label='Cyst')
    plt.xlabel('Time (days)')
    plt.ylabel('Total Population')
    plt.title('Life Stage Dynamics')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Risk assessment (surface concentration)
    plt.subplot(3, 3, 8)
    surface_conc = results['T_profiles'][:, 0]  # Surface trophozoites
    risk_threshold = 1000  # cells/mL
    risk_level = surface_conc / risk_threshold
    
    colors = ['green' if r < 0.1 else 'orange' if r < 1.0 else 'red' for r in risk_level]
    plt.scatter(results['times']/24, risk_level, c=colors, alpha=0.6)
    plt.axhline(1.0, color='red', linestyle='--', label='High Risk Threshold')
    plt.xlabel('Time (days)')
    plt.ylabel('Risk Level (relative to threshold)')
    plt.title('PAM Risk Assessment')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Summary statistics
    plt.subplot(3, 3, 9)
    stats_text = f"""
    MODEL PARAMETERS:
    Max Growth Rate: {model.r_max:.2f} hr⁻¹
    Optimal Temperature: {model.T_opt}°C
    Carrying Capacity: {model.K:.1e} cells/mL
    Water Depth: {model.depth:.1f} m
    
    SIMULATION RESULTS:
    Duration: {results['times'][-1]/24:.1f} days
    Peak Population: {results['total_populations'].max():.2e}
    Final Population: {results['total_populations'][-1]:.2e}
    Growth Factor: {results['total_populations'][-1]/results['total_populations'][1]:.1e}
    
    RISK ASSESSMENT:
    Max Surface Conc.: {results['T_profiles'][:, 0].max():.1e} cells/mL
    Days Above Threshold: {np.sum(results['T_profiles'][:, 0] > 1000):.0f}
    """
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    return fig


def run_parameter_study():
    """
    Run parameter sensitivity study for student analysis.
    """
    print("Running parameter sensitivity study...")
    
    # Base case
    model = SimpleNFowleriModel(depth=3.0, kerala_climate=True)
    model.T[0:3] = 1000  # Initial contamination near surface
    model.C[5:10] = 100  # Some cysts deeper
    
    base_results = model.run_simulation(t_final=240, dt=0.1)  # 10 days
    
    # Parameter variations
    variations = {
        'depth': [1.0, 2.0, 3.0, 4.0, 5.0],
        'r_max': [0.2, 0.3, 0.5, 0.7, 1.0],
        'T_opt': [35, 37, 40, 42, 45]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (param, values) in enumerate(variations.items()):
        final_populations = []
        
        for value in values:
            # Create model with varied parameter
            test_model = SimpleNFowleriModel(depth=3.0, kerala_climate=True)
            setattr(test_model, param, value)
            
            # Reset initial conditions
            test_model.T[0:3] = 1000
            test_model.C[5:10] = 100
            
            # Run simulation
            test_results = test_model.run_simulation(t_final=240, dt=0.1)
            final_populations.append(test_results['total_populations'][-1])
        
        # Plot results
        axes[i].plot(values, final_populations, 'bo-', linewidth=2, markersize=8)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Final Population')
        axes[i].set_title(f'Sensitivity to {param}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    return fig


def create_educational_animation(results, model, filename='nfowleri_animation.gif'):
    """
    Create animation showing evolution of N. fowleri in water column.
    Great for educational presentations!
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    def animate(frame):
        ax1.clear()
        ax2.clear() 
        ax3.clear()
        
        # Extract data for this time frame
        T_profile = results['T_profiles'][frame, :]
        F_profile = results['F_profiles'][frame, :]
        C_profile = results['C_profiles'][frame, :]
        temp_profile = results['temp_profiles'][frame, :]
        t = results['times'][frame]
        
        # Plot 1: Organism concentrations vs depth
        ax1.semilogx(T_profile + 1, model.z, 'r-', linewidth=2, label='Trophozoite')
        ax1.semilogx(F_profile + 1, model.z, 'g-', linewidth=2, label='Flagellate')
        ax1.semilogx(C_profile + 1, model.z, 'b-', linewidth=2, label='Cyst')
        ax1.set_xlabel('Concentration (cells/mL)')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title('Organism Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        ax1.set_xlim([1, 1e6])
        
        # Plot 2: Temperature profile
        ax2.plot(temp_profile, model.z, 'r-', linewidth=3)
        ax2.axvline(model.T_opt, color='g', linestyle='--', alpha=0.7, label='Optimal')
        ax2.axvspan(model.T_min, model.T_max, alpha=0.2, color='orange')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Temperature Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        ax2.set_xlim([20, 45])
        
        # Plot 3: Population time series
        ax3.semilogy(results['times'][:frame+1]/24, results['total_populations'][:frame+1], 
                    'b-', linewidth=2)
        ax3.scatter(t/24, results['total_populations'][frame], color='red', s=50, zorder=5)
        ax3.set_xlabel('Time (days)')
        ax3.set_ylabel('Total Population')
        ax3.set_title('Population Growth')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, results['times'][-1]/24])
        
        # Add time annotation
        fig.suptitle(f'N. fowleri Dynamics in Kerala Pond - Day {t/24:.1f}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
    
    # Create animation
    frames = len(results['times'])
    anim = FuncAnimation(fig, animate, frames=frames, interval=200, repeat=True)
    
    # Save as GIF
    anim.save(filename, writer='pillow', fps=5)
    print(f"Animation saved as {filename}")
    
    return anim


# Main execution for student project
if __name__ == "__main__":
    print("=" * 80)
    print("N. fowleri Growth Model for Tropical Water Bodies")
    print("Student Implementation - Government College Kasaragod")
    print("=" * 80)
    
    # Create model for typical Kerala pond
    print("\n1. Setting up Kerala pond model...")
    model = SimpleNFowleriModel(depth=3.0, kerala_climate=True)
    
    # Set initial conditions (small contamination event)
    print("2. Setting initial contamination...")
    model.T[0:2] = 1000   # Surface contamination (cells/mL)
    model.F[0] = 100      # Some flagellates
    model.C[3:8] = 50     # Cysts in sediment layer
    
    # Run simulation
    print("3. Running simulation...")
    results = model.run_simulation(t_final=360, dt=0.1, save_interval=2.0)  # 15 days
    
    # Plot results
    print("4. Generating plots...")
    fig = plot_simulation_results(results, model)
    plt.savefig('nfowleri_kerala_pond_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Parameter sensitivity study
    print("5. Running parameter sensitivity study...")
    param_fig = run_parameter_study()
    plt.show()
    
    # Create educational animation
    print("6. Creating animation...")
    anim = create_educational_animation(results, model, 'nfowleri_education.gif')
    
    print("\nSimulation complete!")
    print(f"Peak population: {results['total_populations'].max():.2e} cells")
    print(f"Final population: {results['total_populations'][-1]:.2e} cells")
    print(f"Days above risk threshold: {np.sum(results['T_profiles'][:, 0] > 1000)}")
    
    # Suggestions for further study
    print("\n" + "=" * 80)
    print("SUGGESTIONS FOR FURTHER STUDY:")
    print("=" * 80)
    print("1. Vary initial conditions and compare growth patterns")
    print("2. Test effect of monsoon (dilution) on population")
    print("3. Compare stagnant vs flowing water scenarios")
    print("4. Investigate effect of water depth on thermal stratification")
    print("5. Model intervention strategies (chlorination, heating)")
    print("6. Validate with laboratory growth curve data")
    print("7. Add seasonal variation for full annual cycle")
    print("8. Create risk maps for different Kerala districts")
    print("=" * 80)
