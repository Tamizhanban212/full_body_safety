import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np

# Ensure this matches your file name (danger_fields.py)
from danger_field_copy import Dangerfields  

# --- Setup ---
app = dash.Dash(__name__)
df_calculator = Dangerfields()

# Grid settings
# 40x40 is a good balance between resolution and real-time slider speed
resolution = 40
x_range = np.linspace(-1, 3, resolution)
y_range = np.linspace(-1, 3, resolution)

# Robot Link Positions (Fixed for this demo)
p_start = np.array([0.0, 0.0, 0.0])
p_end   = np.array([2.0, 2.0, 0.0])

# --- Layout ---
app.layout = html.Div([
    html.H1("Real-Time Danger Field (Logarithmic)", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
    
    html.Div([
        # Graph Area
        dcc.Graph(id='heatmap-graph', style={'height': '70vh'}),
    ]),

    html.Div([
        # Controls Area
        html.Div([
            html.H4("Velocity Start (v_i)"),
            html.Label("X Component:"),
            dcc.Slider(id='vi-x', min=-2, max=2, step=0.1, value=0.1, 
                       marks={i: str(i) for i in range(-2, 3)}),
            html.Label("Y Component:"),
            dcc.Slider(id='vi-y', min=-2, max=2, step=0.1, value=0.0, 
                       marks={i: str(i) for i in range(-2, 3)}),
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),

        html.Div([
            html.H4("Velocity End (v_ip1)"),
            html.Label("X Component:"),
            dcc.Slider(id='vip1-x', min=-2, max=2, step=0.1, value=0.5, 
                       marks={i: str(i) for i in range(-2, 3)}),
            html.Label("Y Component:"),
            dcc.Slider(id='vip1-y', min=-2, max=2, step=0.1, value=0.5, 
                       marks={i: str(i) for i in range(-2, 3)}),
        ], style={'width': '40%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),
    ], style={'textAlign': 'center', 'backgroundColor': '#f9f9f9', 'padding': '20px'})
])

# --- Callback (The Logic) ---
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('vi-x', 'value'),
     Input('vi-y', 'value'),
     Input('vip1-x', 'value'),
     Input('vip1-y', 'value')]
)
def update_heatmap(vi_x, vi_y, vip1_x, vip1_y):
    # 1. Update Velocity Vectors
    v_start = np.array([float(vi_x), float(vi_y), 0.0])
    v_end   = np.array([float(vip1_x), float(vip1_y), 0.0])

    # 2. Recalculate Grid
    Z_values = np.zeros((resolution, resolution))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            r_point = np.array([x, y, 0.0])
            
            try:
                # Calculate Raw Danger
                val = df_calculator.calculate_cksdf_link(
                    r=r_point, 
                    r_i=p_start, 
                    r_ip1=p_end, 
                    v_i=v_start, 
                    v_ip1=v_end, 
                    k1=1.0, k2=1.0, gamma=1.0 # Standard weights
                )
                
                # --- KEY CHANGE: Logarithmic Scale ---
                # The paper uses log scale because values near the arm are huge.
                # We add 1e-6 to avoid log(0).
                if val < 0: val = 0
                val = np.log10(val + 1e-6)

                # Clip high values (singularity near the arm) for better color contrast
                if val > 2.5: val = 2.5
                if val < -1.5: val = -1.5 # Clip extremely low values

            except Exception as e:
                val = -1.0 # Default background
            
            # Plotly expects Z[y, x]
            Z_values[j, i] = val

    # 3. Create Figure
    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Contour(
        z=Z_values,
        x=x_range,
        y=y_range,
        colorscale='Jet',
        # Adjust contours for Log Scale values (usually between -1 and 3)
        contours=dict(
            start=-1.0, 
            end=2.5, 
            size=0.1, 
            showlines=False
        ),
        colorbar=dict(title='Log10(Danger)')
    ))

    # Robot Link Overlay
    fig.add_trace(go.Scatter(
        x=[p_start[0], p_end[0]],
        y=[p_start[1], p_end[1]],
        mode='lines+markers',
        line=dict(color='white', width=5),
        marker=dict(size=8, color='white'),
        name='Link Segment'
    ))

    # Velocity Arrows
    # Arrow 1: Start
    fig.add_annotation(
        x=p_start[0] + v_start[0], y=p_start[1] + v_start[1],
        xref="x", yref="y",
        ax=p_start[0], ay=p_start[1], axref="x", ayref="y",
        arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor="cyan",
        opacity=0.8
    )
    # Arrow 2: End
    fig.add_annotation(
        x=p_end[0] + v_end[0], y=p_end[1] + v_end[1],
        xref="x", yref="y",
        ax=p_end[0], ay=p_end[1], axref="x", ayref="y",
        arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor="magenta",
        opacity=0.8
    )

    fig.update_layout(
        title=f"Field Distribution (Log Scale)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        margin=dict(l=40, r=40, t=40, b=40),
        transition={'duration': 100} # Smooth animation
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)