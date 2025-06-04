# Central Provident Fund (CPF) Board Manpower Optimisation System

Timeline: January 21, 2024 → April 27, 2024
Client: Central Provident Fund Board
My Role: Solution Engineer
Tools: Agent Based Simulation, Discrete Event Simulation, Figma, Javascript, Pandas library, Python, R Studio, Systems Dynamics
Document: ../Final_Presentation.pdf

### **Manpower Optimisation System for CPF**

**Objective:** Develop a system to optimise manpower allocation for the Central Provident Fund (CPF) Board, addressing the challenge of accurately forecasting enquiry volumes during peak periods.

**Challenge:** CPF faced operational inefficiencies due to inaccurate forecasting of incoming enquiry volumes, leading to over/understaffing and unnecessary operational costs.

**Approach:**

- **Stakeholder Engagement:** Conducted site visits and interviews with key stakeholders to understand pain points.
- **Design Prototyping:** Iterated multiple design prototypes with the client, refining the solution to meet their needs.
- **Manpower Optimisation:** Designed a system to address forecasting challenges and optimise staff allocation.

**Role:**

- Led the design of solutions for each identified pain point, creating Figma prototypes.
- Developed a mathematical model to effectively forecast enquiry demand.
- Researched and selected a suitable simulation system to support the project’s objectives.

## Systems Dynamics Simulation System

---

![Screenshot 2024-08-27 at 5.48.21 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-27_at_5.48.21_PM.png)

## Exploration of other Simulation System

---

- Agent Based Simulation
    
    ![Screenshot 2024-08-28 at 1.07.55 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_1.07.55_PM.png)
    
- Discrete Event Simulation
    
    ![Screenshot 2024-08-28 at 1.08.12 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_1.08.12_PM.png)
    

## Solution Design Iterations

---

- Iteration 1
    
    ![Screenshot 2024-08-28 at 1.13.50 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_1.13.50_PM.png)
    
- Iteration 2
    
    ![Screenshot 2024-08-28 at 1.12.58 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_1.12.58_PM.png)
    
- Final Solution
    
    [Screen Recording 2024-08-28 at 2.30.00 PM.mov](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screen_Recording_2024-08-28_at_2.30.00_PM.mov)
    

## Simulating Open Balance

---

- **Forecasting New Incoming Cases** - With historical data of daily new cases, our machine learning regression model is able to forecast the number of new cases coming in.
    
    ![Screenshot 2024-08-28 at 2.23.49 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_2.23.49_PM.png)
    
- **Simulating Total Closed Case** - With historical data of daily productivity rate for each employee type, we are able to make an accurate prediction of number of daily case closure rate.
    
    ![Screenshot 2024-08-28 at 2.27.12 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_2.27.12_PM.png)
    
- **Simulating Open Balance** - With the above 2 data, we simply simulate open balance of cases for the upcoming week by finding the difference between both!
    
    ![Screenshot 2024-08-28 at 2.28.32 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_2.28.32_PM.png)
    

## Data-driven Manpower Allocation

---

- Using the data, we developed a data-driven approach to guide supervisors on the number of cases each employee type should close to meet their weekly targets.
    
    ![Screenshot 2024-08-28 at 2.34.02 PM.png](Central%20Provident%20Fund%20(CPF)%20Board%20Manpower%20Optimi%20b2d449e2d8584adbaf3efcfdcede232c/Screenshot_2024-08-28_at_2.34.02_PM.png)
    

## Outcome

---

**ed**

**Data-driven approach** to determine daily staffing levels, **improving operational efficiency.**