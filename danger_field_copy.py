import numpy as np

class Dangerfields:
    def __init__(self):
        pass

    def calculate_cksdf_link(self, r, r_i, r_ip1, v_i, v_ip1, k1=1.0, k2=1.0, gamma=1.0):
        """
        Calculates the Danger Field using the Closed-Form Analytical Solution 
        from Lacevic et al.
        
        This method decomposes the velocity integral into three standard forms:
        J0 = Integral( 1 / Q^(3/2) )
        J1 = Integral( t / Q^(3/2) )
        J2 = Integral( t^2 / Q^(3/2) )
        """
        
        # --- 1. Geometric Coefficients ---
        # r_i: start of link, r_ip1: end of link
        # v_i: velocity start, v_ip1: velocity end
        
        diff_r = r_ip1 - r_i
        diff_v = v_ip1 - v_i
        
        # Relative position vector at t=0: Delta_0 = r - r_i
        delta_0 = r - r_i
        
        # Quadratic Distance Coefficients Q(t) = a*t^2 + b*t + c
        # a = ||diff_r||^2
        a = np.dot(diff_r, diff_r)
        # b = -2 * (delta_0 . diff_r)
        b = -2 * np.dot(delta_0, diff_r)
        # c = ||delta_0||^2
        c = np.dot(delta_0, delta_0)

        # --- 2. Velocity Coefficients ---
        # The numerator v(t) . (r - p(t)) expands to M*t^2 + N*t + P
        # Note: The sqrt(||v||) term from the definition cancels out in the projection,
        # leaving pure polynomials.
        
        M = -np.dot(diff_v, diff_r)
        N = np.dot(diff_v, delta_0) - np.dot(v_i, diff_r)
        P = np.dot(v_i, delta_0)

        # --- 3. Solve Integrals (Closed Form) ---
        
        # Safety Check: If 'a' is tiny, link length is zero.
        if a < 1e-9:
            return 0.0

        # Determinant of the quadratic equation: Delta = 4ac - b^2
        D = 4*a*c - b**2
        
        # Safety Check: If D is very small, the point lies exactly on the infinite line of the link.
        if abs(D) < 1e-9:
            return 100.0 # Singularity
            
        # Helper values at boundaries t=0 and t=1
        Q0 = c
        Q1 = a + b + c
        
        # Sqrt values (safeguarded against negative zeros)
        sqrt_Q0 = np.sqrt(Q0) if Q0 > 0 else 1e-9
        sqrt_Q1 = np.sqrt(Q1) if Q1 > 0 else 1e-9

        # --- Integral I_pos: Position Field ---
        # Formula: Int(1/sqrt(Q)) = (1/sqrt(a)) * ln( ... )
        sqrt_a = np.sqrt(a)
        num_log = 2*a + b + 2*sqrt_a*sqrt_Q1
        den_log = b + 2*sqrt_a*sqrt_Q0
        
        # Log safety
        if den_log <= 0: den_log = 1e-9
        if num_log <= 0: num_log = 1e-9
            
        I_pos = (1.0 / sqrt_a) * np.log(num_log / den_log)


        # --- Integral I_vel: Velocity Field ---
        # We need Int( (M*t^2 + N*t + P) / Q^(3/2) )
        # This is M*J2 + N*J1 + P*J0
        
        # J0 = Int(1 / Q^(3/2))
        # Standard solution: 2(2at + b) / (D * sqrt(Q))
        val_J0_1 = (4*a + 2*b) / (D * sqrt_Q1)
        val_J0_0 = (2*b) / (D * sqrt_Q0)
        J0 = val_J0_1 - val_J0_0

        # J1 = Int(t / Q^(3/2))
        # Standard solution: -2(2c + bt) / (D * sqrt(Q))
        val_J1_1 = -2*(2*c + b) / (D * sqrt_Q1)
        val_J1_0 = -2*(2*c) / (D * sqrt_Q0)
        J1 = val_J1_1 - val_J1_0

        # J2 = Int(t^2 / Q^(3/2))
        # Recursive relation allows calculating J2 using I_pos, J1, and J0
        # Int(t^2 / Q^1.5) = (1/a)*Int(1/sqrt(Q)) - (b/2a)*J1 - (c/a)*J0
        J2 = (1.0/a)*I_pos - (b/(2*a))*J1 - (c/a)*J0

        # Combine weighted terms
        I_vel = M*J2 + N*J1 + P*J0

        # --- 4. Final Sum ---
        # The velocity term contributes only if it increases danger.
        # In the strict formulation, negative values (moving away) reduce danger.
        
        total_danger = (k1 * I_pos) + (k2 * gamma * I_vel)
        
        # Clamp to 0 (Danger cannot be negative)
        return max(0.0, total_danger)