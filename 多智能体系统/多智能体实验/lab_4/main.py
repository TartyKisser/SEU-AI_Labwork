import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import random
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False


class Person:
    def __init__(self, x, y, state='S'):
        self.x = x
        self.y = y
        self.state = state  # 'S' (Susceptible), 'I' (Infected), or 'R' (Recovered)
        self.velocity = np.random.uniform(-1, 1, 2)
        self.velocity = self.velocity / max(np.linalg.norm(self.velocity), 0.1)
        self.time_infected = 0
        self.speed = np.random.uniform(0.5, 1.5)
        self.isolation = False

    def move(self, width, height, population, social_distancing_factor=0.0):
        # Random movement
        self.x += self.velocity[0] * self.speed
        self.y += self.velocity[1] * self.speed

        # Boundary check (bounce)
        if self.x <= 0 or self.x >= width:
            self.velocity[0] *= -1
            self.x = max(0, min(self.x, width))

        if self.y <= 0 or self.y >= height:
            self.velocity[1] *= -1
            self.y = max(0, min(self.y, height))

        # Randomly change direction (add natural movement)
        if random.random() < 0.05:
            angle = np.random.uniform(0, 2 * np.pi)
            self.velocity = np.array([np.cos(angle), np.sin(angle)])


class EpidemicSimulation:
    def __init__(self, population_size=200, width=800, height=600,
                 initial_infected=5, infection_radius=10, infection_rate=0.5,
                 recovery_days=14, mortality_rate=0.02):
        self.width = width
        self.height = height
        self.population_size = population_size
        self.day = 0
        self.dt = 0.2  # Time step, represents how many days each update equals

        # Epidemic parameters
        self.infection_radius = infection_radius
        self.infection_rate = infection_rate
        self.recovery_days = recovery_days
        self.recovery_rate = 1.0 / recovery_days  # Recovery rate per day
        self.mortality_rate = mortality_rate
        self.initial_infected = initial_infected

        # Initialize population
        self.reset_population()

        # Statistical data
        self.history = {'S': [self.count_state('S')],
                        'I': [self.count_state('I')],
                        'R': [self.count_state('R')],
                        'D': [0]}  # Death count

    def reset_population(self):
        """Reset population with current parameters"""
        self.population = []

        # Create population
        for i in range(self.population_size):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            self.population.append(Person(x, y, 'S'))

        # Set initial infected individuals
        for i in range(min(self.initial_infected, self.population_size)):
            self.population[i].state = 'I'

    def count_state(self, state):
        return sum(1 for person in self.population if person.state == state)

    def update(self):
        # Update population status
        for person in self.population:
            if person.state == 'S':
                # Check if infected
                for other in self.population:
                    if other.state == 'I':
                        distance = np.sqrt((person.x - other.x) ** 2 + (person.y - other.y) ** 2)
                        if distance < self.infection_radius:
                            # Infection probability depends on distance and transmission rate
                            infection_probability = self.infection_rate * (1 - distance / self.infection_radius)
                            if random.random() < infection_probability:
                                person.state = 'I'
                                break

            elif person.state == 'I':
                # Update infection time
                person.time_infected += self.dt

                # Check if recovered or dead based on recovery rate
                recovery_probability = self.recovery_rate * self.dt
                if random.random() < recovery_probability:
                    if random.random() < self.mortality_rate:
                        person.state = 'D'  # Death
                    else:
                        person.state = 'R'  # Recovery

        # Move population
        for person in self.population:
            if person.state != 'D':  # Dead people don't move
                person.move(self.width, self.height, self.population)

        # Update statistical data
        self.history['S'].append(self.count_state('S'))
        self.history['I'].append(self.count_state('I'))
        self.history['R'].append(self.count_state('R'))
        self.history['D'].append(sum(1 for person in self.population if person.state == 'D'))

        self.day += self.dt

        # Return current number of infected individuals
        return self.count_state('I')


def simulate_epidemic():
    # Create simulation
    population_size = 500
    width, height = 800, 600
    simulation = EpidemicSimulation(
        population_size=population_size,
        width=width,
        height=height,
        initial_infected=5,
        infection_radius=30,
        infection_rate=0.8,
        recovery_days=30,
        mortality_rate=0.1
    )

    # Create a clearer layout
    fig = plt.figure(figsize=(16, 12))  # Increase both width and height

    # Use GridSpec for more precise layout control - increase gap between slider columns
    gs = GridSpec(5, 3, height_ratios=[3, 3, 0.6, 0.6, 0.6], width_ratios=[1.3, 1.3, 0.5],
                  hspace=0.5, wspace=0.6, left=0.12, right=0.98, top=0.95, bottom=0.15)

    # Top left plot - Population distribution
    ax1 = plt.subplot(gs[0:2, 0])
    ax1.set_xlim(0, width)
    ax1.set_ylim(0, height)
    ax1.set_title('Epidemic Transmission Simulation', fontsize=12, pad=10)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')

    # Top right plot - Status change curves
    ax2 = plt.subplot(gs[0:2, 1])
    ax2.set_title('Population Status Changes', fontsize=12, pad=10)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Number of People')

    # Right side - Color legend
    legend_ax = plt.subplot(gs[0:2, 2])
    legend_ax.axis('off')
    legend_ax.text(0.1, 0.9, "Color Legend", fontweight='bold', fontsize=11)
    legend_ax.text(0.1, 0.75, "● Susceptible", color='blue', fontsize=10)
    legend_ax.text(0.1, 0.6, "● Infected", color='red', fontsize=10)
    legend_ax.text(0.1, 0.45, "● Recovered", color='green', fontsize=10)
    legend_ax.text(0.1, 0.3, "● Dead", color='black', fontsize=10)
    legend_ax.text(0.1, 0.1, "Hover over left\nplot to show\ninfection radius",
                   fontsize=9, style='italic')

    # Initialize scatter plot
    colors = []
    for person in simulation.population:
        if person.state == 'S':
            colors.append('blue')
        elif person.state == 'I':
            colors.append('red')
        elif person.state == 'R':
            colors.append('green')
        else:  # 'D'
            colors.append('black')

    scatter = ax1.scatter(
        [person.x for person in simulation.population],
        [person.y for person in simulation.population],
        c=colors, s=30, alpha=0.7
    )

    # Create statistical plot lines
    days = [0]
    line_s, = ax2.plot(days, simulation.history['S'], 'b-', label='Susceptible')
    line_i, = ax2.plot(days, simulation.history['I'], 'r-', label='Infected')
    line_r, = ax2.plot(days, simulation.history['R'], 'g-', label='Recovered')
    line_d, = ax2.plot(days, simulation.history['D'], 'k-', label='Dead')
    ax2.legend()

    # Create dedicated area for sliders with larger gaps between columns
    ax_infection_rate = plt.subplot(gs[2, 0])
    s_infection_rate = Slider(ax_infection_rate, 'Infection Rate', 0.0, 1.0, valinit=simulation.infection_rate)

    ax_infection_radius = plt.subplot(gs[2, 1])
    s_infection_radius = Slider(ax_infection_radius, 'Infection Radius', 5.0, 50.0, valinit=simulation.infection_radius)

    # Second row of sliders
    ax_recovery_rate = plt.subplot(gs[3, 0])
    s_recovery_rate = Slider(ax_recovery_rate, 'Recovery Rate', 0.01, 0.5, valinit=simulation.recovery_rate)

    ax_mortality_rate = plt.subplot(gs[3, 1])
    s_mortality_rate = Slider(ax_mortality_rate, 'Mortality Rate', 0.0, 0.2, valinit=simulation.mortality_rate)

    # Third row of sliders - spanning both columns for initial infected
    ax_initial_infected = plt.subplot(gs[4, 0:2])
    s_initial_infected = Slider(ax_initial_infected, 'Initial Infected Count', 1, 50,
                                valinit=simulation.initial_infected, valfmt='%d')

    # Add control buttons - repositioned for better fit
    reset_button_ax = plt.axes([0.3, 0.05, 0.11, 0.05])
    button_reset = Button(reset_button_ax, 'Reset Parameters')

    pause_button_ax = plt.axes([0.43, 0.05, 0.08, 0.05])
    button_pause = Button(pause_button_ax, 'Pause')

    restart_button_ax = plt.axes([0.53, 0.05, 0.11, 0.05])
    button_restart = Button(restart_button_ax, 'Restart Simulation')

    # Add status information at bottom - adjusted left margin
    status_ax = plt.axes([0.12, 0.01, 0.75, 0.03])
    status_ax.axis('off')
    status_text = status_ax.text(0.5, 0.5, "Simulation Status: Ready to start",
                                 ha='center', va='center', fontsize=10,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    # Infection area display
    infection_area = plt.Circle((0, 0), simulation.infection_radius, color='red', alpha=0.1)
    ax1.add_patch(infection_area)
    infection_area.set_visible(False)

    # Animation control variable
    animation_paused = [False]  # Use list to make it mutable in nested functions

    # Update function
    def update_params(val):
        simulation.infection_rate = s_infection_rate.val
        simulation.infection_radius = s_infection_radius.val
        simulation.recovery_rate = s_recovery_rate.val
        simulation.mortality_rate = s_mortality_rate.val
        new_initial_infected = int(s_initial_infected.val)

        # If initial infected count changed, reset simulation
        if new_initial_infected != simulation.initial_infected:
            simulation.initial_infected = new_initial_infected
            restart_simulation()

        infection_area.set_radius(simulation.infection_radius)

    # Reset function - only resets parameters to default
    def reset(event):
        s_infection_rate.reset()
        s_infection_radius.reset()
        s_recovery_rate.reset()
        s_mortality_rate.reset()
        s_initial_infected.reset()
        update_params(None)
        status_text.set_text("Simulation Status: Parameters reset")

    # Pause/Resume function
    def toggle_pause(event):
        animation_paused[0] = not animation_paused[0]
        if animation_paused[0]:
            button_pause.label.set_text('Resume')
            status_text.set_text(f"Simulation Status: Paused at Day {simulation.day:.1f}")
        else:
            button_pause.label.set_text('Pause')
            status_text.set_text(f"Simulation Status: Running - Day {simulation.day:.1f}")

    # Restart function - restarts the entire simulation
    def restart_simulation():
        animation_paused[0] = False
        button_pause.label.set_text('Pause')
        simulation.day = 0
        simulation.history = {'S': [], 'I': [], 'R': [], 'D': []}
        simulation.reset_population()
        simulation.history['S'].append(simulation.count_state('S'))
        simulation.history['I'].append(simulation.count_state('I'))
        simulation.history['R'].append(simulation.count_state('R'))
        simulation.history['D'].append(0)
        days.clear()
        days.append(0)
        status_text.set_text("Simulation Status: Restarted")

    def restart(event):
        restart_simulation()

    # Register callbacks
    s_infection_rate.on_changed(update_params)
    s_infection_radius.on_changed(update_params)
    s_recovery_rate.on_changed(update_params)
    s_mortality_rate.on_changed(update_params)
    s_initial_infected.on_changed(update_params)
    button_reset.on_clicked(reset)
    button_pause.on_clicked(toggle_pause)
    button_restart.on_clicked(restart)

    # Hover event: show infection radius
    def hover(event):
        if event.inaxes == ax1:
            infection_area.center = (event.xdata, event.ydata)
            infection_area.set_visible(True)
        else:
            infection_area.set_visible(False)
        fig.canvas.draw_idle()

    # Connect events
    fig.canvas.mpl_connect('motion_notify_event', hover)

    # Epidemic transmission animation
    def animate(frame):
        # Skip animation update if paused
        if animation_paused[0]:
            return scatter, line_s, line_i, line_r, line_d

        # Update simulation
        active_cases = simulation.update()

        # Update scatter plot
        colors = []
        for person in simulation.population:
            if person.state == 'S':
                colors.append('blue')
            elif person.state == 'I':
                colors.append('red')
            elif person.state == 'R':
                colors.append('green')
            else:  # 'D'
                colors.append('black')

        scatter.set_offsets([(person.x, person.y) for person in simulation.population])
        scatter.set_color(colors)

        # Update statistical plot
        days.append(simulation.day)
        line_s.set_data(days, simulation.history['S'])
        line_i.set_data(days, simulation.history['I'])
        line_r.set_data(days, simulation.history['R'])
        line_d.set_data(days, simulation.history['D'])

        ax2.relim()
        ax2.autoscale_view()

        # Update titles and status
        ax1.set_title(f'Epidemic Transmission Simulation - Day {simulation.day:.1f}', fontsize=12, pad=10)
        ax2.set_title(f'Population Status Changes (Total: {len(simulation.population)}, Active Cases: {active_cases})',
                      fontsize=12, pad=10)

        if not animation_paused[0]:
            status_text.set_text(
                f"Running - Day {simulation.day:.1f} | S:{simulation.count_state('S')} I:{active_cases} R:{simulation.count_state('R')} D:{simulation.count_state('D')}")

        return scatter, line_s, line_i, line_r, line_d

    # Create animation
    anim = FuncAnimation(fig, animate, frames=500, interval=50, blit=False)

    plt.show()


if __name__ == "__main__":
    simulate_epidemic()