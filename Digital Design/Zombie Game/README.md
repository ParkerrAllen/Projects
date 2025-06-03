
### README: Zombie Apocalypse Simulation

---

## **Zombie Apocalypse Simulation**

This is a 3D zombie survival game implemented using OpenGL. Your objective is to survive for 3 minutes while avoiding zombies. The game includes features like zombie spawning, player health, a game timer, and first-person controls. The game adjusts based on player interactions, such as restarting or losing health.

---

## **Controls**

### **Keyboard Controls**
- **Navigation**:
  - `W`, `A`, `S`, `D`: Move forward, left, backward, and right in first-person mode.
  - Arrow Keys: Adjust the camera in first-person mode.
  - Mouse: To move the camera in either direction.
- **View Modes**:
  - `M`: Cycle between orthogonal, perspective, and first-person views.
  - `+`/`-`: Adjust the fov in perspective mode.
- **Lighting Control**:
  - `V`: Toggle between light and camera control mode.
  - Arrow Keys (in lighting mode): Adjust the position and angle of the light source.
  - `W`, `S`: Move the light source up and down.
- **Reset Game**
  - `R`: Reset Game
  - `Esc`: Exit the game
  

## **How to Run**

1. **Compiling the Code**:
   - Run make

2. **Running the Program**:
   -
     ```bash
     ./final
     ```

## **Key Features to Observe**:

### **Zombie Behavior:**

Zombies dynamically spawn and move toward the player.

Their speed increases as the game progresses.

Watch their animations for smooth joint movements.

### **Player Interactions:**

Health bar decreases when zombies hit the player.

The health bar is displayed at the top-left corner.

### **Game Timer:**

The timer is shown on the top-right corner and counts down from 3 minutes.

When the timer reaches 0, a win screen appears.

### **Game Over/Restart:**

If health reaches 0, the game shows a "Game Over" screen.

Pressing R resets the game state and starts with two zombies.

### **Dynamic Environment:**

Collision detection prevents walking through objects like skyscrapers and rocks.

## **Shortcuts to Key Features**

Win State: Reduce the timer quickly for demonstration by adjusting tLimit to a lower value in the code.

Game Over State: Set playerHealth to 0 to demonstrate the game-over screen.

Restart State: Press R to show the restart functionality.

## **Why I Deserve an A**

This project demonstrates:

Mastery of OpenGL through dynamic rendering, animation, and lighting.

Complex game mechanics including collision detection, zombie AI, and first-person controls.

Thoughtful design with a focus on usability.

Custom textures and assets enhance the visual appeal.

Testing to ensure edge cases.

## **Acknowledgments**

CSCIx229 Library: Base library provided for OpenGL functionality.

Online Tutorials:

OpenGL tutorials for guidance on lighting for street lights and some for camera, which in turn helped me out with other lighting and perspective issues I was having.

Collision detection ideas adapted from various sources, including StackOverflow discussions and reddit. Also for glMatrixMode I also used these sources that taught me that I can use it to pick which matrix is in the current state so I used it for perspectives and the win/lose windows. And I used these sources for mouse issues and help and for the print function. 

Code Reuse
Any reused code has been modified and was used from past homeworks and examples that I did in this class and other classes. I used past examples and homework to help me with things I forgot.

Known Issues

Occasionally, some zombies might get stuck due to collision detection with the environment. I tried preventing this by adding a way to move a zombie.

Restarting the game resets to two zombies, but their positions may occasionally overlap.

The zombies overlap each other, and disappears sometimes.

