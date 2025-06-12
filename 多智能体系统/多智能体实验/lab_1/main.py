import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import random
import matplotlib

matplotlib.use('TkAgg')
from typing import Dict, List, Tuple, Any
import time


class Resource:
    def __init__(self, name: str, total_supply: int, initial_price: float):
        self.name = name
        self.total_supply = total_supply
        self.available_supply = total_supply
        self.price = initial_price
        self.usage_history = []  # Track usage over time for price adjustment
        self.price_history = [initial_price]  # Track price changes
        self.utilization_history = [0]  # Track utilization percentage

    def allocate(self, requested_amount: int, max_price: float) -> bool:
        """Allocate resources if available and price is acceptable"""
        if self.available_supply >= requested_amount and self.price <= max_price:
            self.available_supply -= requested_amount
            return True
        return False

    def release(self, amount: int) -> None:
        """Return resources to the available pool"""
        self.available_supply += amount
        if self.available_supply > self.total_supply:
            self.available_supply = self.total_supply  # Safety check

    def update_price(self) -> float:
        """Update price based on supply and demand"""
        # Calculate utilization rate
        utilization_rate = 1 - (self.available_supply / self.total_supply)

        # Store current utilization for history
        self.usage_history.append(utilization_rate)
        if len(self.usage_history) > 10:
            self.usage_history.pop(0)  # Keep only last 10 time steps

        # Calculate average recent utilization
        avg_utilization = sum(self.usage_history) / len(self.usage_history)

        # Dynamic pricing formula:
        # - High utilization increases price
        # - Low utilization decreases price
        # - Prices never go below 1
        price_change_rate = np.power(1.5, (2 * avg_utilization - 1))
        self.price = max(1, self.price * price_change_rate)

        # Store price and utilization for history
        self.price_history.append(self.price)
        self.utilization_history.append(utilization_rate * 100)  # As percentage

        return self.price

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the resource"""
        return {
            "name": self.name,
            "total": self.total_supply,
            "available": self.available_supply,
            "utilized": self.total_supply - self.available_supply,
            "utilization_rate": ((self.total_supply - self.available_supply) / self.total_supply) * 100,
            "price": self.price
        }


class Task:
    def __init__(self, task_id: int, resource_requirements: Dict[str, List[float]]):
        self.id = task_id
        self.resource_requirements = resource_requirements
        self.is_running = False
        self.completion_time = -1  # When will this task complete
        self.allocated_resources = {}  # Track allocated resources
        self.start_time = -1  # When did this task start

    def can_start(self, resources: Dict[str, Resource], current_time: int) -> bool:
        """Check if the task can start with current resource prices"""
        if self.is_running:
            return False

        # Check if all resource requirements can be met
        for resource_name, requirement in self.resource_requirements.items():
            amount, duration, max_price = requirement
            resource = resources.get(resource_name)

            if not resource or resource.price > max_price or resource.available_supply < amount:
                return False

        return True

    def start(self, resources: Dict[str, Resource], current_time: int) -> bool:
        """Allocate resources and start the task"""
        if self.is_running:
            return False

        # Temporary allocation to check if all resources can be allocated
        temp_allocations = {}

        # Try to allocate all resources
        for resource_name, requirement in self.resource_requirements.items():
            amount, duration, max_price = requirement
            resource = resources.get(resource_name)

            if resource.allocate(amount, max_price):
                temp_allocations[resource_name] = amount
            else:
                # Release any resources already allocated in this attempt
                for release_name, release_amount in temp_allocations.items():
                    resources[release_name].release(release_amount)
                return False

        # All resources allocated successfully
        self.is_running = True
        self.allocated_resources = {}  # Clear previous allocations
        self.start_time = current_time

        # Set completion time and track allocations
        for resource_name, requirement in self.resource_requirements.items():
            amount, duration, max_price = requirement
            self.allocated_resources[resource_name] = {
                "amount": amount,
                "duration": duration
            }

            # Set completion time based on the longest duration
            if self.completion_time == -1 or current_time + duration > self.completion_time:
                self.completion_time = current_time + duration

        return True

    def update(self, resources: Dict[str, Resource], current_time: int) -> bool:
        """Update task state and release resources if completed"""
        if not self.is_running or current_time < self.completion_time:
            return False  # Task is not running or not completed yet

        # Task is completed, release resources
        for resource_name, allocation in self.allocated_resources.items():
            resources[resource_name].release(allocation["amount"])

        # Reset task state
        self.is_running = False
        self.completion_time = -1
        self.allocated_resources = {}

        return True  # Task completed

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the task"""
        return {
            "id": self.id,
            "running": self.is_running,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "allocated_resources": self.allocated_resources,
            "resource_requirements": self.resource_requirements
        }


class MarketBasedResourceSystem:
    def __init__(self):
        # Initialize resources with initial supply and price
        self.resources = {
            "CPU": Resource("CPU", 250, 10),
            "Memory": Resource("Memory", 500, 5),
            "Storage": Resource("Storage", 600, 2),
            "Network": Resource("Network", 200, 8)
        }

        self.tasks = []  # List of all tasks
        self.completed_tasks = []  # Track completed tasks
        self.current_time = 0  # Current time step
        self.history = []  # Track system state over time

        # For visualization
        self.task_start_times = []
        self.task_completion_times = []
        self.running_tasks_history = [0]
        self.pending_tasks_history = [0]
        self.completed_tasks_history = [0]

    def add_task(self, task_resource_requirements: Dict[str, List[float]]) -> int:
        """Add a new task to the system"""
        task_id = len(self.tasks) + 1
        task = Task(task_id, task_resource_requirements)
        self.tasks.append(task)
        return task_id

    def step(self) -> Dict[str, Any]:
        """Process a single time step"""
        self.current_time += 1

        # Record current state before changes
        self.record_history()

        # Update completed tasks and release resources
        newly_completed = []
        for task in self.tasks:
            if task.update(self.resources, self.current_time):
                completed_info = {
                    "id": task.id,
                    "completed_at": self.current_time,
                    "start_time": task.start_time,
                    "duration": self.current_time - task.start_time
                }
                self.completed_tasks.append(completed_info)
                self.task_completion_times.append(self.current_time)
                newly_completed.append(completed_info)

        # Try to start new tasks
        pending_tasks = [task for task in self.tasks if not task.is_running]
        newly_started = []

        for task in pending_tasks:
            if task.can_start(self.resources, self.current_time):
                task.start(self.resources, self.current_time)
                self.task_start_times.append(self.current_time)
                newly_started.append(task.id)

        # Update resource prices based on utilization
        for resource in self.resources.values():
            resource.update_price()

        # Update running/pending task history
        running_count = len([task for task in self.tasks if task.is_running])
        pending_count = len([task for task in self.tasks if not task.is_running])
        self.running_tasks_history.append(running_count)
        self.pending_tasks_history.append(pending_count)
        self.completed_tasks_history.append(len(self.completed_tasks))

        return {
            "time": self.current_time,
            "resource_status": self.get_resource_status(),
            "running_tasks": running_count,
            "pending_tasks": pending_count,
            "completed_tasks": len(self.completed_tasks),
            "newly_started": newly_started,
            "newly_completed": newly_completed
        }

    def run(self, steps: int, task_types, visualize: bool = False, visualize_interval: int = 10) -> List[
        Dict[str, Any]]:
        """Run simulation for specified number of time steps"""
        results = []

        if visualize:
            plt.ion()  # Turn on interactive mode
            fig = self.create_visualization_figure()

        for i in range(steps):

            if random.random() < 0.2:
                task_type_index = i % 4  # Cycle through task types
                self.add_task(task_types[task_type_index])

            step_result = self.step()
            results.append(step_result)

            # Update visualization periodically
            if visualize and (i % visualize_interval == 0 or i == steps - 1):
                self.update_visualization(fig)
                plt.pause(0.01)  # Small pause to update the plots

        if visualize:
            plt.ioff()  # Turn off interactive mode
            plt.show()  # Keep the final plot visible

        return results

    def get_resource_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all resources"""
        status = {}
        for name, resource in self.resources.items():
            status[name] = resource.get_status()
        return status

    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """Get list of currently running tasks"""
        return [task.get_status() for task in self.tasks if task.is_running]

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get list of pending tasks"""
        return [task.get_status() for task in self.tasks if not task.is_running]

    def record_history(self) -> None:
        """Record current system state"""
        self.history.append({
            "time": self.current_time,
            "resources": self.get_resource_status(),
            "running_tasks": len(self.get_running_tasks()),
            "pending_tasks": len(self.get_pending_tasks()),
            "completed_tasks": len(self.completed_tasks)
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        return {
            "current_time": self.current_time,
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "running_tasks": len(self.get_running_tasks()),
            "pending_tasks": len(self.get_pending_tasks()),
            "resource_utilization": [{
                "name": name,
                "utilization": ((resource.total_supply - resource.available_supply) / resource.total_supply) * 100
            } for name, resource in self.resources.items()],
            "prices": [{
                "name": name,
                "price": resource.price
            } for name, resource in self.resources.items()]
        }

    def create_visualization_figure(self) -> plt.Figure:
        """Create a figure for visualization"""
        # Modified to have only 3 plots instead of 4 (removed Task Timeline)
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        # Resource Price subplot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Resource Prices Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price')

        # Resource Utilization subplot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Resource Utilization Over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Utilization (%)')

        # Task Execution subplot
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_title('Task Execution')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Number of Tasks')

        fig.tight_layout()
        return fig

    def update_visualization(self, fig: plt.Figure) -> None:
        """Update the visualization with current data"""
        time_steps = list(range(self.current_time + 1))

        # Clear all subplots
        for ax in fig.axes:
            ax.clear()

        # Resource Price subplot (ax1)
        ax1 = fig.axes[0]
        for name, resource in self.resources.items():
            # Ensure price history has same length as time_steps
            prices = resource.price_history + [resource.price_history[-1]] * (
                        len(time_steps) - len(resource.price_history))
            ax1.plot(time_steps, prices, label=f"{name} Price")
        ax1.set_title('Resource Prices Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Resource Utilization subplot (ax2)
        ax2 = fig.axes[1]
        for name, resource in self.resources.items():
            # Ensure utilization history has same length as time_steps
            utilization = resource.utilization_history + [resource.utilization_history[-1]] * (
                        len(time_steps) - len(resource.utilization_history))
            ax2.plot(time_steps, utilization, label=f"{name} Utilization")
        ax2.set_title('Resource Utilization Over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend()

        # Task Execution subplot (ax3)
        ax3 = fig.axes[2]
        ax3.plot(time_steps, self.running_tasks_history, label="Running Tasks")
        ax3.plot(time_steps, self.pending_tasks_history, label="Pending Tasks")
        ax3.plot(time_steps, self.completed_tasks_history, label="Completed Tasks")
        ax3.set_title('Task Execution Status')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Number of Tasks')
        ax3.legend()

        # Update the figure
        fig.tight_layout()

    def visualize_final_results(self) -> None:
        """Create a final visualization of the simulation results"""
        # Modified to remove the Task Timeline subplot
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(3, 2, figure=fig)  # Changed from 4 to 3 rows

        time_steps = list(range(self.current_time + 1))

        # Resource Price subplot
        ax1 = fig.add_subplot(gs[0, 0])
        for name, resource in self.resources.items():
            # Ensure price history has same length as time_steps
            prices = resource.price_history + [resource.price_history[-1]] * (
                        len(time_steps) - len(resource.price_history))
            ax1.plot(time_steps, prices, label=f"{name} Price")
        ax1.set_title('Resource Prices Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Price')
        ax1.legend()

        # Resource Utilization subplot
        ax2 = fig.add_subplot(gs[0, 1])
        for name, resource in self.resources.items():
            # Ensure utilization history has same length as time_steps
            utilization = resource.utilization_history + [resource.utilization_history[-1]] * (
                        len(time_steps) - len(resource.utilization_history))
            ax2.plot(time_steps, utilization, label=f"{name} Utilization")
        ax2.set_title('Resource Utilization Over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Utilization (%)')
        ax2.legend()

        # Task Execution subplot
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(time_steps, self.running_tasks_history, label="Running Tasks")
        ax3.plot(time_steps, self.pending_tasks_history, label="Pending Tasks")
        ax3.plot(time_steps, self.completed_tasks_history, label="Completed Tasks")
        ax3.set_title('Task Execution Status')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Number of Tasks')
        ax3.legend()

        # Final resource state
        ax5 = fig.add_subplot(gs[2, 0])
        resource_names = list(self.resources.keys())
        final_prices = [resource.price for resource in self.resources.values()]
        ax5.bar(resource_names, final_prices)
        ax5.set_title('Final Resource Prices')
        ax5.set_ylabel('Price')

        # Task completion statistics
        ax6 = fig.add_subplot(gs[2, 1])
        stats = [len(self.tasks), len(self.completed_tasks),
                 len([task for task in self.tasks if task.is_running]),
                 len([task for task in self.tasks if not task.is_running])]
        labels = ['Total Tasks', 'Completed Tasks', 'Running Tasks', 'Pending Tasks']
        ax6.bar(labels, stats)
        ax6.set_title('Task Statistics')
        ax6.set_ylabel('Count')

        plt.tight_layout()
        plt.show()

    def save_data_to_csv(self, filename: str) -> None:
        """Save simulation data to CSV for external analysis"""
        # Resource price and utilization history
        resource_data = []

        for t in range(self.current_time + 1):
            row = {"Time": t}

            for name, resource in self.resources.items():
                # Price data
                if t < len(resource.price_history):
                    row[f"{name}_Price"] = resource.price_history[t]
                else:
                    row[f"{name}_Price"] = resource.price_history[-1]

                # Utilization data
                if t < len(resource.utilization_history):
                    row[f"{name}_Utilization"] = resource.utilization_history[t]
                else:
                    row[f"{name}_Utilization"] = resource.utilization_history[-1]

            resource_data.append(row)

        # Task data
        task_data = []

        for completed in self.completed_tasks:
            task_data.append({
                "Task_ID": completed["id"],
                "Start_Time": completed["start_time"],
                "Completion_Time": completed["completed_at"],
                "Duration": completed["duration"]
            })

        # Running tasks
        for task in self.tasks:
            if task.is_running:
                status = task.get_status()
                task_data.append({
                    "Task_ID": status["id"],
                    "Start_Time": status["start_time"],
                    "Completion_Time": "Running",
                    "Duration": self.current_time - status["start_time"]
                })

        # Save to CSV
        pd.DataFrame(resource_data).to_csv(f"{filename}_resources.csv", index=False)
        pd.DataFrame(task_data).to_csv(f"{filename}_tasks.csv", index=False)


def run_simulation():
    """Run a sample simulation and visualize the results"""
    # Create system
    system = MarketBasedResourceSystem()

    # Define 4 task types with different resource requirements
    task_types = [
        # Task Type 1: CPU-intensive
        {
            "CPU": [20, 20, 30],  # [amount, duration, maxPrice]
            "Memory": [10, 20, 15],
            "Storage": [5, 20, 10],
            "Network": [2, 20, 20]
        },
        # Task Type 2: Memory-intensive
        {
            "CPU": [5, 15, 20],
            "Memory": [40, 15, 25],
            "Storage": [10, 15, 10],
            "Network": [5, 15, 15]
        },
        # Task Type 3: Storage-intensive
        {
            "CPU": [10, 25, 25],
            "Memory": [15, 25, 20],
            "Storage": [50, 25, 30],
            "Network": [8, 25, 20]
        },
        # Task Type 4: Network-intensive
        {
            "CPU": [8, 10, 20],
            "Memory": [16, 10, 15],
            "Storage": [15, 10, 10],
            "Network": [25, 10, 40]
        }
    ]

    # Add a batch of tasks (mix of different types)
    for i in range(20):
        task_type_index = i % 4  # Cycle through task types
        system.add_task(task_types[task_type_index])

    print("Starting simulation...")

    # Run simulation for 100 time steps with real-time visualization
    # We'll update the visualization every 5 steps for performance
    results = system.run(100, task_types, visualize=True, visualize_interval=5)

    # Create and show final visualization
    system.visualize_final_results()

    # Print final statistics
    print("\n=== Simulation Results ===")
    stats = system.get_statistics()
    print(f"Time steps: {stats['current_time']}")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Completed tasks: {stats['completed_tasks']}")
    print(f"Running tasks: {stats['running_tasks']}")
    print(f"Pending tasks: {stats['pending_tasks']}")

    # Print final resource prices
    print("\n=== Final Resource Prices ===")
    for price_info in stats['prices']:
        print(f"{price_info['name']}: {price_info['price']:.2f}")

    # Print resource utilization
    print("\n=== Resource Utilization ===")
    for util_info in stats['resource_utilization']:
        print(f"{util_info['name']}: {util_info['utilization']:.2f}%")

    # Save data for external analysis
    system.save_data_to_csv("simulation_results")

    return system, results


if __name__ == "__main__":
    # Run the simulation
    system, results = run_simulation()