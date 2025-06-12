# -*- coding: utf-8 -*-
import pygame
import numpy as np
import sys
from pygame.locals import *

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (240, 240, 240)
TEXT_COLOR = (50, 50, 50)
BOID_COLOR = (65, 105, 225)  # Blue
VISION_RADIUS_COLOR = (200, 200, 200, 50)  # Vision radius color
SEPARATION_RADIUS_COLOR = (255, 100, 100, 50)  # Separation radius color
TRAIL_COLOR = (100, 100, 255, 20)  # Trail color
SLIDER_BACKGROUND = (200, 200, 200)
SLIDER_FOREGROUND = (100, 150, 220)
SLIDER_HANDLE = (80, 120, 200)
BUTTON_COLOR = (100, 150, 220)
BUTTON_HOVER_COLOR = (80, 120, 200)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Boids Model Simulation')
clock = pygame.time.Clock()

# Font setup
font = pygame.font.Font(None, 24)


class Vector2D:
    """2D vector class for position, velocity and acceleration"""
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        if scalar == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return self / mag
    
    def limit(self, max_val):
        mag = self.magnitude()
        if mag > max_val:
            return self.normalize() * max_val
        return Vector2D(self.x, self.y)
    
    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    @staticmethod
    def random():
        """Returns a random directional unit vector"""
        angle = np.random.random() * 2 * np.pi
        return Vector2D(np.cos(angle), np.sin(angle))


class Slider:
    """Slider UI component"""
    
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, step=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.step = step
        self.dragging = False
        
        # Calculate initial position of the handle
        self.handle_pos = self.value_to_pos(initial_val)
    
    def value_to_pos(self, value):
        """Convert a value to a position on the slider"""
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return int(self.x + ratio * self.width)
    
    def pos_to_value(self, pos):
        """Convert a position to a value"""
        ratio = (pos - self.x) / self.width
        value = self.min_val + ratio * (self.max_val - self.min_val)
        
        # Apply step if provided
        if self.step:
            value = round(value / self.step) * self.step
        
        return max(self.min_val, min(self.max_val, value))
    
    def handle_event(self, event):
        """Handle mouse events for the slider"""
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_x, mouse_y = event.pos
                # Check if click is on the handle
                handle_rect = pygame.Rect(self.handle_pos - 5, self.y - 5, 10, 10)
                if handle_rect.collidepoint(mouse_x, mouse_y):
                    self.dragging = True
                # Or on the slider itself
                slider_rect = pygame.Rect(self.x, self.y - 5, self.width, 10)
                if slider_rect.collidepoint(mouse_x, mouse_y):
                    self.handle_pos = mouse_x
                    self.value = self.pos_to_value(mouse_x)
                    self.dragging = True
        
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                self.dragging = False
        
        elif event.type == MOUSEMOTION:
            if self.dragging:
                mouse_x, mouse_y = event.pos
                self.handle_pos = max(self.x, min(self.x + self.width, mouse_x))
                self.value = self.pos_to_value(self.handle_pos)
    
    def draw(self, surface):
        """Draw the slider"""
        # Draw label
        label_surface = font.render(f"{self.label}: {self.value:.1f}", True, TEXT_COLOR)
        surface.blit(label_surface, (self.x, self.y - 25))
        
        # Draw slider background
        pygame.draw.rect(surface, SLIDER_BACKGROUND, (self.x, self.y - 2, self.width, 4))
        
        # Draw slider foreground (filled part)
        filled_width = self.handle_pos - self.x
        pygame.draw.rect(surface, SLIDER_FOREGROUND, (self.x, self.y - 2, filled_width, 4))
        
        # Draw handle
        pygame.draw.circle(surface, SLIDER_HANDLE, (self.handle_pos, self.y), 8)


class Checkbox:
    """Checkbox UI component"""
    
    def __init__(self, x, y, width, height, label, checked=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.checked = checked
        self.rect = pygame.Rect(x, y, width, height)
    
    def handle_event(self, event):
        """Handle mouse events for the checkbox"""
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if self.rect.collidepoint(event.pos):
                    self.checked = not self.checked
                    return True
        return False
    
    def draw(self, surface):
        """Draw the checkbox"""
        # Draw box
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, 2)
        
        # Draw checkmark if checked
        if self.checked:
            inner_rect = pygame.Rect(self.x + 4, self.y + 4, self.width - 8, self.height - 8)
            pygame.draw.rect(surface, SLIDER_FOREGROUND, inner_rect)
        
        # Draw label
        label_surface = font.render(self.label, True, TEXT_COLOR)
        surface.blit(label_surface, (self.x + self.width + 10, self.y + self.height/2 - 10))


class Button:
    """Button UI component"""
    
    def __init__(self, x, y, width, height, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.rect = pygame.Rect(x, y, width, height)
        self.hovered = False
    
    def handle_event(self, event):
        """Handle mouse events for the button"""
        if event.type == MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if self.rect.collidepoint(event.pos):
                    return True
        return False
    
    def draw(self, surface):
        """Draw the button"""
        color = BUTTON_HOVER_COLOR if self.hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, TEXT_COLOR, self.rect, 2)
        
        # Draw label
        label_surface = font.render(self.label, True, BUTTON_TEXT_COLOR)
        label_rect = label_surface.get_rect(center=(self.x + self.width/2, self.y + self.height/2))
        surface.blit(label_surface, label_rect)


class Boid:
    """Boid class represents an individual in the flock"""
    
    def __init__(self, x, y, params):
        self.position = Vector2D(x, y)
        self.velocity = Vector2D.random() * (params['min_speed'] + np.random.random() * 
                                            (params['max_speed'] - params['min_speed']))
        self.acceleration = Vector2D(0, 0)
        self.params = params
        self.color = tuple(np.random.randint(50, 220, 3))  # Random color
        self.trail = []  # Trail
        self.max_trail_length = 20
    
    def apply_behavior(self, boids):
        """Apply all behavior rules"""
        separation = self.separate(boids)
        alignment = self.align(boids)
        cohesion = self.cohere(boids)
        
        # Apply weights
        separation = separation * self.params['separation_weight']
        alignment = alignment * self.params['alignment_weight']
        cohesion = cohesion * self.params['cohesion_weight']
        
        # Accumulate acceleration
        self.acceleration = self.acceleration + separation
        self.acceleration = self.acceleration + alignment
        self.acceleration = self.acceleration + cohesion
        
        # Add a bit of randomness
        if np.random.random() < 0.05:
            jitter = Vector2D.random() * 0.1
            self.acceleration = self.acceleration + jitter
    
    def separate(self, boids):
        """Separation rule - avoid collisions with other boids"""
        steering = Vector2D(0, 0)
        count = 0
        
        for other in boids:
            if other is self:
                continue
            
            distance = self.position.distance_to(other.position)
            
            if 0 < distance < self.params['separation_radius']:
                # Calculate avoidance vector
                diff = self.position - other.position
                diff = diff.normalize()
                # Greater influence for closer boids
                diff = diff / distance
                steering = steering + diff
                count += 1
        
        if count > 0:
            steering = steering / count
        
        if steering.magnitude() > 0:
            steering = steering.normalize() * self.params['max_speed']
            steering = steering - self.velocity
            steering = steering.limit(self.params['max_force'])
        
        return steering
    
    def align(self, boids):
        """Alignment rule - match direction with nearby boids"""
        steering = Vector2D(0, 0)
        count = 0
        
        for other in boids:
            if other is self:
                continue
            
            distance = self.position.distance_to(other.position)
            
            if 0 < distance < self.params['perception_radius']:
                steering = steering + other.velocity
                count += 1
        
        if count > 0:
            steering = steering / count
            steering = steering.normalize() * self.params['max_speed']
            steering = steering - self.velocity
            steering = steering.limit(self.params['max_force'])
        
        return steering
    
    def cohere(self, boids):
        """Cohesion rule - move toward the center of nearby boids"""
        steering = Vector2D(0, 0)
        count = 0
        
        for other in boids:
            if other is self:
                continue
            
            distance = self.position.distance_to(other.position)
            
            if 0 < distance < self.params['perception_radius']:
                steering = steering + other.position
                count += 1
        
        if count > 0:
            steering = steering / count
            return self.seek(steering)
        
        return steering
    
    def seek(self, target):
        """Helper function to seek a target point"""
        desired = target - self.position
        desired = desired.normalize() * self.params['max_speed']
        
        steer = desired - self.velocity
        steer = steer.limit(self.params['max_force'])
        
        return steer
    
    def update(self):
        """Update position and velocity"""
        # Update velocity
        self.velocity = self.velocity + self.acceleration
        
        # Limit speed
        speed = self.velocity.magnitude()
        if speed > self.params['max_speed']:
            self.velocity = self.velocity.normalize() * self.params['max_speed']
        elif speed < self.params['min_speed']:
            self.velocity = self.velocity.normalize() * self.params['min_speed']
        
        # Update position
        self.position = self.position + self.velocity
        
        # Record trail
        if self.params['show_trails']:
            self.trail.append((self.position.x, self.position.y))
            if len(self.trail) > self.max_trail_length:
                self.trail.pop(0)
        else:
            self.trail = []
        
        # Reset acceleration
        self.acceleration = Vector2D(0, 0)
        
        # Handle boundaries
        self.handle_boundary()
    
    def handle_boundary(self):
        """Boundary handling - wrap around screen"""
        if self.position.x < 0:
            self.position.x = WIDTH
        elif self.position.x > WIDTH:
            self.position.x = 0
            
        if self.position.y < 0:
            self.position.y = HEIGHT
        elif self.position.y > HEIGHT:
            self.position.y = 0
    
    def draw(self, surface):
        """Draw the boid and its visual effects"""
        # Draw perception radius and separation radius
        if self.params['show_radius']:
            # Create temporary surface for translucent circles
            temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            
            # Draw perception radius
            pygame.draw.circle(
                temp_surface, 
                VISION_RADIUS_COLOR, 
                (int(self.position.x), int(self.position.y)), 
                self.params['perception_radius'], 
                1
            )
            
            # Draw separation radius
            pygame.draw.circle(
                temp_surface, 
                SEPARATION_RADIUS_COLOR, 
                (int(self.position.x), int(self.position.y)), 
                self.params['separation_radius'], 
                1
            )
            
            surface.blit(temp_surface, (0, 0))
        
        # Draw trail
        if self.params['show_trails'] and len(self.trail) > 1:
            # Create temporary surface for translucent trail
            temp_trail = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            
            # Use recorded trail points
            if len(self.trail) >= 2:
                pygame.draw.lines(
                    temp_trail, 
                    (*self.color, 100),  # Use boid color but translucent
                    False, 
                    self.trail,
                    1
                )
                
            surface.blit(temp_trail, (0, 0))
        
        # Calculate direction angle
        angle = np.arctan2(self.velocity.y, self.velocity.x)
        
        # Draw triangle (representing the boid)
        boid_points = [
            (self.position.x + 10 * np.cos(angle), 
             self.position.y + 10 * np.sin(angle)),
            (self.position.x + 5 * np.cos(angle + 2.5), 
             self.position.y + 5 * np.sin(angle + 2.5)),
            (self.position.x + 5 * np.cos(angle - 2.5), 
             self.position.y + 5 * np.sin(angle - 2.5))
        ]
        
        pygame.draw.polygon(surface, self.color, boid_points)


class Simulation:
    """Simulation manager class"""
    
    def __init__(self):
        self.params = {
            'num_boids': 100,
            'max_speed': 4.0,
            'min_speed': 1.0,
            'max_force': 0.3,
            'separation_weight': 1.5,
            'alignment_weight': 1.0,
            'cohesion_weight': 1.0,
            'perception_radius': 50,
            'separation_radius': 25,
            'show_radius': False,
            'show_trails': False
        }
        
        self.boids = []
        self.init_boids()
        
        # Setup UI controls
        self.setup_ui()
        
        # State variables
        self.paused = False
        self.show_controls = True
        self.last_time = pygame.time.get_ticks()
        self.fps = 60
    
    def init_boids(self):
        """Initialize boid population"""
        self.boids = []
        for _ in range(self.params['num_boids']):
            x = np.random.random() * WIDTH
            y = np.random.random() * HEIGHT
            self.boids.append(Boid(x, y, self.params))
    
    def setup_ui(self):
        """Setup UI control elements"""
        self.controls = {
            # Sliders
            'num_boids': Slider(WIDTH - 240, 50, 180, 10, 10, 300, self.params['num_boids'], "Number of Boids", 10),
            'max_speed': Slider(WIDTH - 240, 90, 180, 10, 1.0, 10.0, self.params['max_speed'], "Max Speed", 0.1),
            'min_speed': Slider(WIDTH - 240, 130, 180, 10, 0.1, 3.0, self.params['min_speed'], "Min Speed", 0.1),
            'separation_weight': Slider(WIDTH - 240, 170, 180, 10, 0.0, 5.0, self.params['separation_weight'], "Separation Weight", 0.1),
            'alignment_weight': Slider(WIDTH - 240, 210, 180, 10, 0.0, 5.0, self.params['alignment_weight'], "Alignment Weight", 0.1),
            'cohesion_weight': Slider(WIDTH - 240, 250, 180, 10, 0.0, 5.0, self.params['cohesion_weight'], "Cohesion Weight", 0.1),
            'perception_radius': Slider(WIDTH - 240, 290, 180, 10, 10, 200, self.params['perception_radius'], "Perception Radius", 5),
            'separation_radius': Slider(WIDTH - 240, 330, 180, 10, 5, 100, self.params['separation_radius'], "Separation Radius", 5),
            
            # Checkboxes
            'show_radius': Checkbox(WIDTH - 240, 370, 20, 20, "Show Perception Radius", self.params['show_radius']),
            'show_trails': Checkbox(WIDTH - 240, 400, 20, 20, "Show Trails", self.params['show_trails']),
            
            # Buttons
            'reset': Button(WIDTH - 240, 440, 85, 30, "Reset"),
            'pause': Button(WIDTH - 145, 440, 85, 30, "Pause")
        }
    
    def handle_events(self):
        """Process events"""
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            
            elif event.type == KEYDOWN:
                # Basic control
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                
                # Toggle controls visibility with H key
                elif event.key == K_h:
                    self.show_controls = not self.show_controls
            
            # Handle UI control events
            if self.show_controls:
                # Slider events
                for key, slider in self.controls.items():
                    if isinstance(slider, Slider):
                        slider.handle_event(event)
                        self.params[key] = slider.value
                        
                        # Special cases
                        if key == 'num_boids' and event.type == MOUSEBUTTONUP:
                            self.init_boids()
                
                # Checkbox events
                for key, checkbox in self.controls.items():
                    if isinstance(checkbox, Checkbox):
                        if checkbox.handle_event(event):
                            self.params[key] = checkbox.checked
                
                # Button events
                if isinstance(self.controls['reset'], Button) and self.controls['reset'].handle_event(event):
                    self.init_boids()
                
                if isinstance(self.controls['pause'], Button) and self.controls['pause'].handle_event(event):
                    self.paused = not self.paused
                    self.controls['pause'].label = "Resume" if self.paused else "Pause"
    
    def update(self):
        """Update simulation state"""
        if not self.paused:
            # Apply behavior rules
            for boid in self.boids:
                boid.apply_behavior(self.boids)
            
            # Update positions
            for boid in self.boids:
                boid.update()
        
        # Update constraint - make sure separation_radius <= perception_radius
        if self.params['separation_radius'] > self.params['perception_radius']:
            self.params['separation_radius'] = self.params['perception_radius']
            self.controls['separation_radius'].value = self.params['separation_radius']
    
    def draw(self):
        """Draw simulation state"""
        # Fill background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw boids
        for boid in self.boids:
            boid.draw(screen)
        
        # Draw UI controls
        if self.show_controls:
            # Draw panel background
            pygame.draw.rect(
                screen, 
                (220, 220, 220, 200), 
                (WIDTH - 250, 40, 240, 440)
            )
            
            # Draw all controls
            for control in self.controls.values():
                control.draw(screen)
        
        # Draw status info
        fps_text = font.render(f"FPS: {self.fps:.1f}", True, TEXT_COLOR)
        status_text = font.render(f"Status: {'Paused' if self.paused else 'Running'}", True, TEXT_COLOR)
        boid_count_text = font.render(f"Boids: {len(self.boids)}", True, TEXT_COLOR)
        
        if not self.show_controls:
            screen.blit(fps_text, (10, 10))
            screen.blit(status_text, (10, 35))
            screen.blit(boid_count_text, (10, 60))
        
        # Update screen
        pygame.display.flip()
    
    def run(self):
        """Run simulation"""
        while True:
            # Calculate FPS
            current_time = pygame.time.get_ticks()
            dt = current_time - self.last_time
            if dt > 0:
                self.fps = 1000 / dt
            self.last_time = current_time
            
            # Process events
            self.handle_events()
            
            # Update
            self.update()
            
            # Draw
            self.draw()
            
            # Control framerate
            clock.tick(60)


# Main program
if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()