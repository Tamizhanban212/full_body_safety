import jax
import jax.numpy as jnp
import numpy as np

# Enable 64-bit precision for stable gradients
jax.config.update("jax_enable_x64", True)

class DangerFields:
    def __init__(self, k1=1.0, k2=0.5, gamma=1.0, discretization=100):
        self.k1 = k1
        self.k2 = k2
        self.gamma = gamma
        self.discretization = discretization

        def total_danger_func(r, v_r, r_starts, r_ends, v_starts, v_ends, k1, k2, gamma):
            link_dangers = jax.vmap(
                self._link_math_dynamic, 
                in_axes=(None, None, 0, 0, 0, 0, None, None, None)
            )(r, v_r, r_starts, r_ends, v_starts, v_ends, k1, k2, gamma)
            
            return jnp.sum(link_dangers)

        self._fast_total_scalar = jax.jit(total_danger_func)
        self._fast_total_grad = jax.jit(jax.grad(total_danger_func, argnums=0))
        
        self._fast_total_scalar_batch = jax.jit(jax.vmap(
            total_danger_func,
            in_axes=(0, 0, None, None, None, None, None, None, None) 
        ))

    @staticmethod
    def _link_math_dynamic(r, v_r, r_i, r_ip1, v_i, v_ip1, k1, k2, gamma):
        """
        Pure math for A SINGLE LINK with RELATIVE VELOCITY.
        """
        num_points = 100 
        s = jnp.linspace(0.0, 1.0, num_points)
        
        r_s = r_i + jnp.outer(s, (r_ip1 - r_i))
        v_s = v_i + jnp.outer(s, (v_ip1 - v_i))

        vec_dist = r - r_s
        v_rel = v_s - v_r  
        
        # --- SAFE DISTANCE CALCULATION ---
        rho_sq = jnp.sum(vec_dist**2, axis=1)
        rho_sq = jnp.maximum(rho_sq, 1e-4) # Prevents division by zero on direct impact
        rho = jnp.sqrt(rho_sq)

        # --- SAFE VELOCITY NORM (THE FIX) ---
        # jnp.linalg.norm(0) yields NaN gradients. We manually calculate the 
        # norm and add a 1e-9 buffer to ensure it never hits absolute zero.
        v_rel_sq = jnp.sum(v_rel**2, axis=1)
        v_rel_sq = jnp.maximum(v_rel_sq, 1e-9) 
        v_rel_mag = jnp.sqrt(v_rel_sq)
        
        interaction = jnp.sum(vec_dist * v_rel, axis=1)

        term1 = k1 * (1.0 / rho)
        term2 = k2 * gamma * (v_rel_mag / rho_sq)
        term3 = k2 * (interaction / (rho_sq * rho))

        return jnp.trapezoid(term1 + term2 + term3, s)
    
    def compute_robot_danger(self, r_target, v_target, r_starts, r_ends, v_starts, v_ends):
        """
        Computes the total danger field for the entire manipulator.
        """
        r = jnp.array(r_target)
        v_r = jnp.array(v_target)
        r_s, r_e = jnp.array(r_starts), jnp.array(r_ends)
        v_s, v_e = jnp.array(v_starts), jnp.array(v_ends)

        total_danger = self._fast_total_scalar(
            r, v_r, r_s, r_e, v_s, v_e, self.k1, self.k2, self.gamma
        )

        total_grad = self._fast_total_grad(
            r, v_r, r_s, r_e, v_s, v_e, self.k1, self.k2, self.gamma
        )

        grad_norm = jnp.linalg.norm(total_grad)
        if grad_norm < 1e-9:
            control_vec = jnp.zeros(3)
        else:
            control_vec = total_danger * (total_grad / grad_norm)

        return float(total_danger), list(total_grad), list(control_vec)

    def compute_robot_danger_batch(self, r_targets, v_targets, r_starts, r_ends, v_starts, v_ends):
        """
        Computes the total danger field for a BATCH of points (e.g., a grid).
        """
        r = jnp.array(r_targets)
        v_r = jnp.array(v_targets) 
        r_s, r_e = jnp.array(r_starts), jnp.array(r_ends)
        v_s, v_e = jnp.array(v_starts), jnp.array(v_ends)

        danger_batch = self._fast_total_scalar_batch(
            r, v_r, r_s, r_e, v_s, v_e, self.k1, self.k2, self.gamma
        )
        return np.array(danger_batch)