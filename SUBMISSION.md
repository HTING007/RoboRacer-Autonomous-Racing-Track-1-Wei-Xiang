# DeepSpeed RoboRacer Hackathon - Track 1 Submission

## Team Members

| Name | Matric Number | Email |
| --- | --- | --- |
| Wei Xiang | TO FILL | TO FILL |
| Lee Jia Han | TO FILL | TO FILL |
| Ting Hong Zhang | TO FILL | HTING007@e.ntu.edu.sg |
| Other member(s) | TO FILL | TO FILL |

## Track

Track 1 - Autonomous Racing

## Approach

Our final solution uses an adaptive Pure Pursuit path-tracking controller with a precomputed racing route.

The controller includes:

- A map-derived shortcut racing route
- Adaptive Pure Pursuit lookahead
- Curvature-based corner-speed planning
- Backward braking constraint propagation
- Forward acceleration constraint propagation
- Steering rate limiting
- Cross-track-error-based speed reduction
- Follow-the-Gap fallback if the precomputed route cannot be loaded

We systematically tested route selection and controller parameters using the provided simulator and evaluator.

A major improvement came from changing from the centreline route to:

`icra26_shortest_clear45.csv`

We also found that controller parameters interact with each other. Increasing the lateral acceleration limit improved lap time but could reduce stability. Increasing the Pure Pursuit lookahead slightly helped recover stability while preserving most of the speed improvement.

The final submitted configuration uses:

- `route_file = icra26_shortest_clear45.csv`
- `a_lat_max = 8.6 m/s^2`
- `lookahead_ratio = 0.34`
- `a_brake = 5.8 m/s^2`
- `a_accel = 5.0 m/s^2`
- `steering_rate_limit = 3.2 rad/s`

These parameters have also been made the default configuration in the submitted code.

## Inputs Used

The controller uses:

- `/ego_racecar/odom`
  - Vehicle position
  - Vehicle heading
  - Vehicle velocity
- `/scan`
  - LiDAR data, primarily used by the fallback Follow-the-Gap controller
- Precomputed route data derived from the supplied track/map

The controller publishes speed and steering commands through `/drive`.

## Results

### Best Verified Result

Result file:

`results/submitted/hz_lat86_L034__20261017T085642Z_r2.json`

- Status: COMPLETE
- Laps completed: 10/10
- Collisions: 0
- Best lap: **19.6712 s**
- Best lap number: 9
- 10-lap total time: **196.7598 s**
- Collision penalty: 0 s
- Real-time factor: 1.817

All ten timed laps were approximately 19.67 s and completed without collisions.

### Default-Configuration Reproducibility Check

We also changed the final parameters to the default submitted configuration and ran the evaluator without supplying a custom driver parameter file.

Result file:

`results/submitted/final_submission_test__20261018T151818Z.json`

- Status: COMPLETE
- Laps completed: 10/10
- Collisions: 0
- Best lap: **19.960 s**
- 10-lap total time: **199.626 s**

This confirms that the submitted repository can run the intended final configuration without requiring a separate experimental parameter file.

## Machine Used for Testing

| Item | Specification |
| --- | --- |
| CPU | AMD Ryzen 5 7530U with Radeon Graphics |
| GPU | AMD Radeon (TM) Graphics |
| RAM | 16 GB |
| Host OS | Windows with WSL2 |
| WSL Distribution | Ubuntu 26.04.1 LTS |
| Container Runtime | Docker Desktop |
| Best-run real-time factor | 1.817 |

## Precomputed Data

The submitted solution uses the precomputed racing route:

`race_ws/src/team_driver/config/icra26_shortest_clear45.csv`

The controller loads this route at startup and computes:

- Segment lengths
- Local route curvature
- Curvature-limited speed profile
- Backward braking constraints
- Forward acceleration constraints

No machine-learning model or externally trained model is used.

## Dependencies Added

No major external runtime dependencies were added beyond the ROS 2 / Python packages available in the provided hackathon environment.

The implementation uses:

- NumPy
- ROS 2 / rclpy
- ackermann_msgs
- nav_msgs
- sensor_msgs
- visualization_msgs
- ament_index_python

## Third-Party Code / References

The solution was developed using the DeepSpeed RoboRacer Hackathon starter repository and simulator environment.

The path-tracking controller is based on the standard Pure Pursuit control concept. The implementation, parameter tuning, route integration, speed-profile logic, and evaluation were adapted and tested by the team for this hackathon.

No externally trained driving model was used.

## AI Assistance

AI tools, including ChatGPT, were used to:

- Help understand the provided codebase and ROS 2 environment
- Troubleshoot WSL2, Docker and Git issues
- Explain controller parameters and Pure Pursuit behaviour
- Suggest structured parameter-tuning experiments
- Analyse interactions between parameters and simulation results
- Assist with debugging and documentation

AI-generated suggestions were tested using the official simulator/evaluator, compared quantitatively, and reviewed before being retained.

Team members are able to explain the final controller and its parameter choices.

## Known Issues / Limitations

The controller becomes less stable if the lateral acceleration limit is increased too aggressively, especially at tight hairpin sections.

Higher-speed configurations can reduce lap time but increase collision risk. Therefore, the final submitted setup prioritises a reproducible zero-collision configuration rather than the fastest unstable configuration.

Performance can also vary slightly between simulation runs depending on the real-time factor of the host machine.
