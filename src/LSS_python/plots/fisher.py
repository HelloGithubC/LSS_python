import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse

from LSS_python.cov import _compute_ellipse_params_from_fisher

def plot_ellipse_from_fisher(fisher, best_fit, ax=None, plot_engine='Ellipse', **kwargs):
    """
    Plot error ellipse from Fisher matrix.

    The error ellipse represents the 1\sigma (68.3%) confidence region
    for two parameters, derived from the Fisher information matrix.

    Parameters
    ----------
    fisher : ndarray
        Fisher information matrix, must be 2x2.
    best_fit : array_like
        Best-fit parameter values [p1, p2] for the ellipse center.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, creates a new figure and axis.
    plot_engine : str, default 'Ellipse'
        Method to draw the ellipse:
        - 'Ellipse': Use matplotlib.patches.Ellipse (default). 
          Note: When axis aspect ratio is not 'equal', the ellipse angle 
          is interpreted in screen space, which may cause misalignment 
          with data-space calculations.
        - 'parametric': Use parametric equation to draw ellipse in data space.
          This ensures correct alignment regardless of aspect ratio.
    **kwargs : dict
        Additional keyword arguments for customization. Supported keys:

        **Ellipse properties** (passed to matplotlib.patches.Ellipse or plot):
        - ellipse_color / color : str, default 'C0'
            Color of the ellipse edge.
        - ellipse_alpha / alpha : float, default 0.5
            Transparency of the ellipse fill.
        - ellipse_facecolor / facecolor : str, optional
            Fill color of the ellipse. If None, uses ellipse_color with alpha.
        - ellipse_edgecolor / edgecolor : str, optional
            Edge color of the ellipse. If None, uses ellipse_color.
        - ellipse_linewidth / linewidth : float, default 1.5
            Width of the ellipse edge.
        - ellipse_linestyle / linestyle : str, default '-'
            Style of the ellipse edge.
        - ellipse_fill / fill : bool, default True
            Whether to fill the ellipse.
        - ellipse_zorder : float, optional
            Z-order for the ellipse patch.
        - n_points : int, default 200
            Number of points for parametric ellipse (only used when plot_engine='parametric').

        **Confidence level**:
        - confidence_level : float, default 0.683
            Confidence level for the ellipse (0 < confidence_level < 1).
            For 1\sigma Gaussian: 0.683, for 2\sigma: 0.954, for 3\sigma: 0.997.
            The ellipse size scales with sqrt(χ² quantile).

        **Center marker**:
        - show_center : bool, default True
            Whether to show a marker at the best-fit center point.
        - center_marker : str, default 'x'
            Marker style for the center point.
        - center_color / center_markercolor : str, default 'C3'
            Color of the center marker.
        - center_size / markersize : float, default 8
            Size of the center marker.
        - center_zorder : float, optional
            Z-order for the center marker.

        **Axis limits**:
        - xlim : tuple, optional
            (xmin, xmax) for the axis. Auto-determined if not provided.
        - ylim : tuple, optional
            (ymin, ymax) for the axis. Auto-determined if not provided.
        - padding : float, default 0.1
            Fractional padding for auto axis limits (10% of ellipse extent).
        - visual_limit : bool, default True
            If True, use visually pleasing axis limits (different ranges for x and y).
            If False, enforce equal axis ranges and aspect ratio to ensure perfect
            alignment between ellipse and sigma points.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axis with the ellipse plotted.

    Raises
    ------
    ValueError
        If fisher is not a 2x2 matrix, or plot_engine is not 'Ellipse' or 'parametric'.
    ImportError
        If matplotlib is not installed.

    Notes
    -----
    The ellipse represents the contour of constant likelihood deviation:
        (θ - θ₀)^T · F · (θ - θ₀) = Δχ²

    For a 1\sigma confidence region in 2D, Δχ² = χ²_{2, 0.683} ≈ 2.28.
    The semi-axis lengths are: a = sqrt(Δχ² / λ₁), b = sqrt(Δχ² / λ₂)
    where λ₁, λ₂ are the eigenvalues of the Fisher matrix.
    """
    import numpy as np

    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError:
        raise ImportError("matplotlib is required for plotting. "
                         "Install it with: pip install matplotlib")

    # Validate fisher matrix dimensions
    fisher = np.atleast_2d(fisher)
    if fisher.shape != (2, 2):
        raise ValueError(f"Fisher matrix must be 2x2 for ellipse plotting, "
                        f"got shape {fisher.shape}")

    # Validate best_fit
    best_fit = np.atleast_1d(best_fit)
    if len(best_fit) != 2:
        raise ValueError(f"best_fit must have exactly 2 elements, got {len(best_fit)}")

    # Validate plot_engine
    if plot_engine not in ['Ellipse', 'parametric']:
        raise ValueError(f"plot_engine must be 'Ellipse' or 'parametric', got '{plot_engine}'")

    # Create axis if not provided
    if ax is None:
        ax = plt.gca()

    # Extract kwargs with defaults
    # Ellipse appearance
    ellipse_color = kwargs.get('ellipse_color', kwargs.get('color', 'C0'))
    ellipse_alpha = kwargs.get('ellipse_alpha', kwargs.get('alpha', 0.5))
    ellipse_facecolor = kwargs.get('ellipse_facecolor', kwargs.get('facecolor', None))
    ellipse_edgecolor = kwargs.get('ellipse_edgecolor', kwargs.get('edgecolor', None))
    ellipse_linewidth = kwargs.get('ellipse_linewidth', kwargs.get('linewidth', 1.5))
    ellipse_linestyle = kwargs.get('ellipse_linestyle', kwargs.get('linestyle', '-'))
    ellipse_fill = kwargs.get('ellipse_fill', kwargs.get('fill', True))
    ellipse_zorder = kwargs.get('ellipse_zorder', kwargs.get('zorder', 1))

    # Confidence level
    confidence_level = kwargs.get('confidence_level', 0.683)

    # Center marker
    show_center = kwargs.get('show_center', True)
    center_marker = kwargs.get('center_marker', 'x')
    center_color = kwargs.get('center_color', kwargs.get('center_markercolor', 'C3'))
    center_size = kwargs.get('center_size', kwargs.get('markersize', 8))
    center_zorder = kwargs.get('center_zorder', 2)

    # Axis limits
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    padding = kwargs.get('padding', 0.1)
    visual_limit = kwargs.get('visual_limit', True)

    # Parametric ellipse parameters
    n_points = kwargs.get('n_points', 200)

    # Compute ellipse parameters using helper function
    ellipse_params = _compute_ellipse_params_from_fisher(fisher, confidence_level)
    
    # Extract parameters
    a = ellipse_params['semi_minor']  # semi-minor axis
    b = ellipse_params['semi_major']  # semi-major axis
    angle_rad = ellipse_params['angle_rad']
    angle_deg = ellipse_params['angle_deg']
    eigenvecs = ellipse_params['eigenvecs']

    # Set facecolor default to ellipse_color with alpha if not specified
    if ellipse_facecolor is None:
        ellipse_facecolor = ellipse_color
    if ellipse_edgecolor is None:
        ellipse_edgecolor = ellipse_color
    
    # Handle fill=False: set facecolor to 'none' instead of using alpha=0
    # This ensures the edge remains visible
    if not ellipse_fill:
        ellipse_facecolor = 'none'

    # Draw ellipse based on plot_engine
    if plot_engine == 'Ellipse':
        # Use matplotlib.patches.Ellipse
        # Note: When aspect ratio is not 'equal', angle is interpreted in screen space
        ellipse = Ellipse(
            xy=best_fit,
            width=2 * a,      # full width (2 * semi-axis)
            height=2 * b,     # full height (2 * semi-axis)
            angle=angle_deg,
            facecolor=ellipse_facecolor,
            edgecolor=ellipse_edgecolor,
            alpha=ellipse_alpha,
            linewidth=ellipse_linewidth,
            linestyle=ellipse_linestyle,
            zorder=ellipse_zorder
        )
        ax.add_patch(ellipse)
    
    else:  # plot_engine == 'parametric'
        # Use parametric equation to draw ellipse in data space
        # This ensures correct alignment regardless of aspect ratio
        
        # Generate parametric points
        theta = np.linspace(0, 2 * np.pi, n_points)
        
        # Ellipse in unrotated coordinate system
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)
        
        # Apply rotation matrix
        # R = [[cos(φ), -sin(φ)], [sin(φ), cos(φ)]]
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        x_rotated = x_ellipse * cos_angle - y_ellipse * sin_angle
        y_rotated = x_ellipse * sin_angle + y_ellipse * cos_angle
        
        # Translate to center
        x_final = x_rotated + best_fit[0]
        y_final = y_rotated + best_fit[1]
        
        # Draw filled ellipse if requested
        if ellipse_fill:
            ax.fill(x_final, y_final,
                   facecolor=ellipse_facecolor,
                   alpha=ellipse_alpha,
                   zorder=ellipse_zorder)
        
        # Draw ellipse edge
        ax.plot(x_final, y_final,
               color=ellipse_edgecolor,
               linewidth=ellipse_linewidth,
               linestyle=ellipse_linestyle,
               zorder=ellipse_zorder)

    # Plot center point if requested
    if show_center:
        ax.plot(best_fit[0], best_fit[1],
                marker=center_marker,
                color=center_color,
                markersize=center_size,
                zorder=center_zorder,
                linestyle='None')

    # Compute bounding box of ellipse for axis limits
    # For a rotated ellipse, the bounding box extent is:
    # width = 2 * sqrt((a*cos(θ))² + (b*sin(θ))²)
    # height = 2 * sqrt((a*sin(θ))² + (b*cos(θ))²)
    # where θ is the rotation angle, a and b are semi-axes
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    half_width = np.sqrt((a * cos_angle)**2 + (b * sin_angle)**2)
    half_height = np.sqrt((a * sin_angle)**2 + (b * cos_angle)**2)
    
    if visual_limit:
        # Visual mode: use visually pleasing axis limits (different ranges)
        # This allows different axis ranges for better visual appearance
        
        # Determine axis limits based on ellipse bounding box or user-provided limits
        if xlim is not None and ylim is not None:
            # Both limits provided by user
            x_min, x_max = xlim
            y_min, y_max = ylim
        elif xlim is not None:
            # Only xlim provided, set ylim based on ellipse
            x_min, x_max = xlim
            y_min = best_fit[1] - half_height
            y_max = best_fit[1] + half_height
        elif ylim is not None:
            # Only ylim provided, set xlim based on ellipse
            y_min, y_max = ylim
            x_min = best_fit[0] - half_width
            x_max = best_fit[0] + half_width
        else:
            # Neither provided, use ellipse bounding box
            x_min = best_fit[0] - half_width
            x_max = best_fit[0] + half_width
            y_min = best_fit[1] - half_height
            y_max = best_fit[1] + half_height
        
        # Calculate ranges and apply padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = padding * max(x_range, 1e-8)
        y_pad = padding * max(y_range, 1e-8)
        
        # Set axis limits independently
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Note: aspect ratio is not forced, allowing visual flexibility
    
    else:
        # Precise alignment mode: enforce equal axis ranges
        # This ensures perfect alignment between ellipse and sigma points
        
        # Determine axis limits based on ellipse bounding box or user-provided limits
        if xlim is not None and ylim is not None:
            # Both limits provided by user
            x_min, x_max = xlim
            y_min, y_max = ylim
        elif xlim is not None:
            # Only xlim provided, set ylim based on ellipse
            x_min, x_max = xlim
            y_min = best_fit[1] - half_height
            y_max = best_fit[1] + half_height
        elif ylim is not None:
            # Only ylim provided, set xlim based on ellipse
            y_min, y_max = ylim
            x_min = best_fit[0] - half_width
            x_max = best_fit[0] + half_width
        else:
            # Neither provided, use ellipse bounding box
            x_min = best_fit[0] - half_width
            x_max = best_fit[0] + half_width
            y_min = best_fit[1] - half_height
            y_max = best_fit[1] + half_height
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Apply padding
        x_pad = padding * max(x_range, 1e-8)
        y_pad = padding * max(y_range, 1e-8)
        
        x_min_padded = x_min - x_pad
        x_max_padded = x_max + x_pad
        y_min_padded = y_min - y_pad
        y_max_padded = y_max + y_pad
        
        # Ensure equal range in both axes to prevent visual distortion
        x_range_padded = x_max_padded - x_min_padded
        y_range_padded = y_max_padded - y_min_padded
        max_range = max(x_range_padded, y_range_padded)
        
        # Center the axis with smaller range
        x_center = (x_min_padded + x_max_padded) / 2
        y_center = (y_min_padded + y_max_padded) / 2
        
        x_min_final = x_center - max_range / 2
        x_max_final = x_center + max_range / 2
        y_min_final = y_center - max_range / 2
        y_max_final = y_center + max_range / 2
        
        # Set axis limits
        ax.set_xlim(x_min_final, x_max_final)
        ax.set_ylim(y_min_final, y_max_final)
        
        # Force equal aspect ratio to ensure ellipse is displayed correctly
        ax.set_aspect('equal')

    # Set labels if not already set
    if ax.get_xlabel() == '':
        ax.set_xlabel('Parameter 1')
    if ax.get_ylabel() == '':
        ax.set_ylabel('Parameter 2')

    return ax