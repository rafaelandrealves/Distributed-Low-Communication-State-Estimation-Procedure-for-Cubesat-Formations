
<h1 align="center">Novel Low-Communication Distributed State Estimation Framework for Satellite Formations 🔭🛰</h1>
<h3 align="center">Collaboration between CMU and IST</h3>


<h3 align="left">Division:</h3>

- **Distributed Consider EKF for a 4 Cubesat Formation w/ Deputies having Relative Range Meas. and a Chief with GPS Meas. - Includes Dynamics with Gravity,J2 Perturbations and Spacecraft Drag with an Exponential Atmosphere model.**
- **Event-Trigger variant of the Proposed Distributed Filter for Sensor Selection and Communication Channel Optimization based on the Crámer-Rao Lower Bound and proportional Confidence Levels, that the user defines to balance the ratio between accuracy and communication optimization.**


<p align="left">
</p>

<h3 align="left">Key Facts:</h3>

- **Satellite formation orbiting Earth in the different stages : (A) - Formation has both relative sensing between every satellite taken by each deputy and GPS measurements from the chief; (B) - Only relative-range being used by deputies; (C) - No measurements are being actuated and the formation graph only contains communication nodes between spacecraft.**

![Screenshot](https://raw.githubusercontent.com/rafaelandrealves/Distributed-Low-Communication-State-Estimation-Procedure-for-Cubesat-Formations/master/Images/Image_ConsEKF.png)

- **The Dynamics Model is made by the author w/ J2 and Drag ➡️ Model includes also the jacobians;**
- **Proposed Distributed Framework is easy to use and has, through testing with CubeSats from the SpaceX Transporter-1 mission, less memory transferred by factor of 1000 when compared to the current literature of distributed systems and 80% less sensor activations that the literature consensus-based distributed frameworks;**

![Screenshot](https://raw.githubusercontent.com/rafaelandrealves/Distributed-Low-Communication-State-Estimation-Procedure-for-Cubesat-Formations/master/Images/MemoryTransferred.png)
![Screenshot](https://raw.githubusercontent.com/rafaelandrealves/Distributed-Low-Communication-State-Estimation-Procedure-for-Cubesat-Formations/master/Images/DecenBarChartAndPieGraph.png)


<p align="left">
</p>


<h3 align="left">Key People:</h3>


- **This work was made by a collaborative effort from the Robotic Exploration Lab at Carnegie-Mellon University and the Institute for Systems and Robotics at Instituto Superiro Técnico;**



<p align="left">
</p>



<p align="left"> <a href="http://roboticexplorationlab.org" target="_blank" rel="noreferrer"> <img src="http://roboticexplorationlab.org/img/logo@2x.png" alt="matlab" width="80" height="80"/> </a> </p>

<p align="left"> <a href="https://www.cmu.edu" target="_blank" rel="noreferrer"> <img src="https://www.cmu.edu/brand/brand-guidelines/images/wordmarksquare-red-600x600.png" alt="matlab1" width="80" height="80"/> </a> </p>

<p align="left"> <a href="https://tecnico.ulisboa.pt/en/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/pt/e/ed/IST_Logo.png" alt="matlab2" width="80" height="80"/> </a> </p>
