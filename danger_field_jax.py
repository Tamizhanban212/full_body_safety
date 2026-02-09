import jax
import jax.numpy as jnp


# Enable 64-bit precision for stable gradients
jax.config.update("jax_enable_x64", True)

class DangerFields:
    def __init__(self, k1=1.0, k2=0.5, gamma=1.0, discretization=100):
        self.k1 = k1
        self.k2 = k2
        self.gamma = gamma
        self.discretization = discretization

        # --- JAX Compilation Setup ---
        # 1. Define the function that computes TOTAL danger (Sum of all links)
        # We use vmap to process N links in parallel, then sum the result.
        def total_danger_func(r, r_starts, r_ends, v_starts, v_ends, k1, k2, gamma):
            # vmap over the link dimensions (axis 0 of r_starts, etc.)
            link_dangers = jax.vmap(
                self._link_math_static, 
                in_axes=(None, 0, 0, 0, 0, None, None, None)
            )(r, r_starts, r_ends, v_starts, v_ends, k1, k2, gamma)
            
            return jnp.sum(link_dangers)

        # 2. JIT Compile the Scalar and Gradient functions
        self._fast_total_scalar = jax.jit(total_danger_func)
        self._fast_total_grad = jax.jit(jax.grad(total_danger_func, argnums=0))

    @staticmethod
    def _link_math_static(r, r_i, r_ip1, v_i, v_ip1, k1, k2, gamma):
        """
        Pure math for A SINGLE LINK.
        (Discretization is hardcoded or passed via closure to keep vmap simple)
        """
        # Hardcoding num_points here for vmap compatibility, 
        # or pass it as a non-traced arg if needed.
        num_points = 100 
        
        s = jnp.linspace(0.0, 1.0, num_points)
        
        # Outer product: (points x 1) * (1 x 3) -> (points x 3)
        # Note: input r_i is shape (3,), so we need to ensure shapes align
        r_s = r_i + jnp.outer(s, (r_ip1 - r_i))
        v_s = v_i + jnp.outer(s, (v_ip1 - v_i))

        vec_dist = r - r_s
        
        # Row-wise dot products
        rho_sq = jnp.sum(vec_dist**2, axis=1)
        rho_sq = jnp.maximum(rho_sq, 1e-9) 
        rho = jnp.sqrt(rho_sq)

        v_mag = jnp.linalg.norm(v_s, axis=1)
        
        # Interaction (Batch Dot Product)
        # Using simple sum(*) is often faster/cleaner than einsum for this specific shape
        interaction = jnp.sum(vec_dist * v_s, axis=1)

        term1 = k1 * (1.0 / rho)
        term2 = k2 * gamma * (v_mag / rho_sq)
        term3 = k2 * (interaction / (rho_sq * rho))

        return jnp.trapezoid(term1 + term2 + term3, s)
    
    def compute_robot_danger(self, r_target, r_starts, r_ends, v_starts, v_ends):
        """
        Computes the total danger field for the entire manipulator.
        
        Args:
            r_target: [x, y, z] point of interest
            r_starts: Numpy array (N_links, 3) 
            r_ends:   Numpy array (N_links, 3)
            v_starts: Numpy array (N_links, 3)
            v_ends:   Numpy array (N_links, 3)
        """
        # Ensure JAX arrays
        r = jnp.array(r_target)
        r_s, r_e = jnp.array(r_starts), jnp.array(r_ends)
        v_s, v_e = jnp.array(v_starts), jnp.array(v_ends)

        # 1. Compute Total Scalar (Sum of all links)
        total_danger = self._fast_total_scalar(
            r, r_s, r_e, v_s, v_e, self.k1, self.k2, self.gamma
        )

        # 2. Compute Total Gradient (Gradient of the sum)
        total_grad = self._fast_total_grad(
            r, r_s, r_e, v_s, v_e, self.k1, self.k2, self.gamma
        )

        # 3. Compute Control Vector
        grad_norm = jnp.linalg.norm(total_grad)
        if grad_norm < 1e-9:
            control_vec = jnp.zeros(3)
        else:
            control_vec = total_danger * (total_grad / grad_norm)

        return float(total_danger), list(total_grad), list(control_vec)

