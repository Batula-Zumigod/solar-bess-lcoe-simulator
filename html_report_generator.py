"""
HTML Report Generator V5
Sections:
  1. Project description (location + load stats)
  2. Summary row (configs, ranges, discount rate)
  3. 3-D scatter  PV × BESS × LCOE coloured by LCOE
  4. 2-D heatmaps with GLOBAL colour scale across all grid limits
     – Renewable fraction, Curtailment, BESS cycles/year
  5. Optimal config card
  6. Discounted cumulative cost line (CAPEX + PV of annual OPEX)
  7. Battery capacity-over-time line (20-year degradation)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def generate_html_report(
    all_configs_df,
    project_name="Solar + BESS Analysis",
    discount_rate=None,
    optimal_capex=None,
    optimal_opex=None,
    load_hourly_data=None,
    optimal_sim_results=None,
    latitude=None,
    longitude=None,
    backup_type=None,
    diesel_escalation=0.0,
    grid_escalation=0.0,
):
    """
    Parameters
    ----------
    all_configs_df      : pd.DataFrame  – all feasible configurations
    project_name        : str
    discount_rate       : float          – e.g. 0.08
    optimal_capex       : dict           – capex_params of optimal config
    optimal_opex        : dict           – opex_params of optimal config
    load_hourly_data    : pd.DataFrame  – must have 'Load_kW' column, 8760 rows
    optimal_sim_results : dict           – full simulate_multi_year() return for optimal
    latitude            : float
    longitude           : float
    backup_type         : str
    diesel_escalation   : float
    grid_escalation     : float

    Returns
    -------
    bytes  (UTF-8 encoded HTML)
    """

    if all_configs_df is None or all_configs_df.empty:
        raise ValueError("all_configs_df cannot be empty")

    df = all_configs_df.copy()

    # ── normalise column names ──
    _renames = {
        'LCOE_Total_/kWh':  'LCOE_Total_$/kWh',
        'LCOE_PV_/kWh':     'LCOE_PV_$/kWh',
        'LCOE_BESS_/kWh':   'LCOE_BESS_$/kWh',
        'LCOE_Backup_/kWh': 'LCOE_Backup_$/kWh',
    }
    df.rename(columns={k: v for k, v in _renames.items() if k in df.columns}, inplace=True)

    required = ['PV_kWp', 'BESS_kWh', 'Grid_Limit_kW', 'LCOE_Total_$/kWh', 'Renewable_Fraction_%']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── derived columns ──
    if 'Curtailment_Rate_%' not in df.columns:
        if 'Curtailed_Energy_MWh' in df.columns and 'PV_Energy_MWh' in df.columns:
            df['Curtailment_Rate_%'] = np.where(
                df['PV_Energy_MWh'] > 0.1,   # >100 kWh threshold, avoids div-by-near-zero
                (df['Curtailed_Energy_MWh'] / df['PV_Energy_MWh'] * 100).clip(0, 100),
                0.0
            )
        else:
            df['Curtailment_Rate_%'] = 0.0

    if 'Generator_Spillage_MWh' not in df.columns:
        df['Generator_Spillage_MWh'] = 0.0

    # Generator spillage rate as % of total diesel generation (to make it comparable across sizes)
    if 'Generator_Spillage_Rate_%' not in df.columns:
        if 'Backup_Energy_MWh' in df.columns:
            df['Generator_Spillage_Rate_%'] = np.where(
                df['Backup_Energy_MWh'] > 0.1,
                (df['Generator_Spillage_MWh'] / df['Backup_Energy_MWh'] * 100).clip(0, 100),
                0.0
            )
        else:
            df['Generator_Spillage_Rate_%'] = 0.0

    if 'BESS_Cycles_per_Year' not in df.columns:
        if 'BESS_Cycle_Count' in df.columns:
            # BESS_Cycle_Count is total over project life; divide by simulation years
            sim_years = len(optimal_sim_results['annual_results']) if optimal_sim_results else 20
            df['BESS_Cycles_per_Year'] = df['BESS_Cycle_Count'] / sim_years
        else:
            df['BESS_Cycles_per_Year'] = 0.0

    df['Curtailment_Rate_%'] = df['Curtailment_Rate_%'].fillna(0)
    df['BESS_Cycles_per_Year'] = df['BESS_Cycles_per_Year'].fillna(0)

    # ── best config ──
    best_idx    = df['LCOE_Total_$/kWh'].idxmin()
    best        = df.loc[best_idx]
    grid_limits = sorted(df['Grid_Limit_kW'].unique())
    timestamp   = datetime.now().strftime('%B %d, %Y  %H:%M')
    dr_pct      = f"{discount_rate*100:.1f}%" if discount_rate is not None else "N/A"

    # ── load statistics ──
    load_avg_kw  = load_peak_kw = load_annual_mwh = 0.0
    if load_hourly_data is not None and isinstance(load_hourly_data, pd.DataFrame):
        col = next((c for c in ['Load_kW', 'load_kw', 'load'] if c in load_hourly_data.columns), None)
        if col:
            arr = load_hourly_data[col].values[:8760].astype(float)
            load_avg_kw    = float(arr.mean())
            load_peak_kw   = float(arr.max())
            load_annual_mwh = float(arr.sum() / 1000)

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 1 – 3-D SCATTER  PV × BESS × LCOE
    # ══════════════════════════════════════════════════════════════════════════
    fig3d = go.Figure()

    # all configs (excluding optimal)
    mask_not_best = df.index != best_idx
    fig3d.add_trace(go.Scatter3d(
        x=df.loc[mask_not_best, 'PV_kWp'],
        y=df.loc[mask_not_best, 'BESS_kWh'],
        z=df.loc[mask_not_best, 'LCOE_Total_$/kWh'],
        mode='markers',
        marker=dict(
            size=4,
            color=df.loc[mask_not_best, 'LCOE_Total_$/kWh'],
            colorscale='RdYlGn',
            reversescale=True,
            colorbar=dict(title='LCOE ($/kWh)', x=1.0, thickness=18),
            opacity=0.75,
            line=dict(width=0),
        ),
        text=[
            f"PV: {r['PV_kWp']:.0f} kWp<br>BESS: {r['BESS_kWh']:.0f} kWh<br>"
            f"Grid: {r['Grid_Limit_kW']:.0f} kW<br>LCOE: ${r['LCOE_Total_$/kWh']:.4f}/kWh"
            for _, r in df.loc[mask_not_best].iterrows()
        ],
        hovertemplate='%{text}<extra></extra>',
        name='Configurations',
    ))

    # optimal star
    fig3d.add_trace(go.Scatter3d(
        x=[best['PV_kWp']],
        y=[best['BESS_kWh']],
        z=[best['LCOE_Total_$/kWh']],
        mode='markers+text',
        marker=dict(size=12, color='gold', symbol='diamond',
                    line=dict(color='#8B0000', width=2)),
        text=['★ Optimal'],
        textposition='top center',
        textfont=dict(size=14, color='#8B0000'),
        hovertemplate=(
            f"<b>★ OPTIMAL</b><br>"
            f"PV: {best['PV_kWp']:.0f} kWp<br>"
            f"BESS: {best['BESS_kWh']:.0f} kWh<br>"
            f"Grid: {best['Grid_Limit_kW']:.0f} kW<br>"
            f"LCOE: ${best['LCOE_Total_$/kWh']:.4f}/kWh<extra></extra>"
        ),
        name='Optimal',
    ))

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(title='PV Capacity (kWp)', backgroundcolor='#f8f9fa',
                       gridcolor='#dee2e6', showbackground=True),
            yaxis=dict(title='BESS Capacity (kWh)', backgroundcolor='#f8f9fa',
                       gridcolor='#dee2e6', showbackground=True),
            zaxis=dict(title='LCOE ($/kWh)', backgroundcolor='#f8f9fa',
                       gridcolor='#dee2e6', showbackground=True),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
        ),
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#ffffff',
        font=dict(family='DM Sans, sans-serif', size=12, color='#2c3e50'),
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
        title=dict(text='PV × BESS × LCOE  — All Configurations',
                   font=dict(size=18), x=0.5, xanchor='center'),
    )
    html_3d = fig3d.to_html(include_plotlyjs=False, div_id='plot_3d', full_html=False)

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 2-4 – 2-D HEATMAPS  (global colour scale per metric)
    # ══════════════════════════════════════════════════════════════════════════

    def _global_heatmap(metric, label, colorscale, reverse, fmt='.2f'):
        """
        Dropdown-controlled 2-D heatmap with a FIXED global colour axis
        so every grid-limit panel uses the same colour mapping.
        """
        vmin = float(df[metric].min())
        vmax = float(df[metric].max())
        # add 5 % padding so extreme points don't hit the very ends of the scale
        pad  = max((vmax - vmin) * 0.05, 1e-9)
        vmin -= pad
        vmax += pad

        fig = go.Figure()
        buttons = []

        for idx, gl in enumerate(grid_limits):
            gdf = df[df['Grid_Limit_kW'] == gl].copy()
            pvs  = sorted(gdf['PV_kWp'].unique())
            bess = sorted(gdf['BESS_kWh'].unique())
            Z    = np.full((len(bess), len(pvs)), np.nan)

            for i, b in enumerate(bess):
                for j, p in enumerate(pvs):
                    m = gdf[(gdf['PV_kWp'] == p) & (gdf['BESS_kWh'] == b)]
                    if len(m):
                        Z[i, j] = m[metric].iloc[0]

            # hover text for each cell
            hover = np.full(Z.shape, '', dtype=object)
            for i, b in enumerate(bess):
                for j, p in enumerate(pvs):
                    v = Z[i, j]
                    val_str = f"{v:{fmt}}" if not np.isnan(v) else "N/A"
                    hover[i, j] = (
                        f"PV: {p:.0f} kWp<br>"
                        f"BESS: {b:.0f} kWh<br>"
                        f"Grid: {gl:.0f} kW<br>"
                        f"{label}: {val_str}"
                    )

            fig.add_trace(go.Heatmap(
                x=pvs, y=bess, z=Z,
                colorscale=colorscale,
                reversescale=reverse,
                zmin=vmin, zmax=vmax,          # ← fixed global scale
                colorbar=dict(
                    title=dict(text=label, side='right'),
                    thickness=18, len=0.8, x=1.02,
                ),
                text=hover,
                hovertemplate='%{text}<extra></extra>',
                visible=(idx == 0),
                name=f'Grid {gl:.0f} kW',
            ))

            # optimal marker (only for its grid limit)
            if best['Grid_Limit_kW'] == gl:
                fig.add_trace(go.Scatter(
                    x=[best['PV_kWp']], y=[best['BESS_kWh']],
                    mode='markers+text',
                    marker=dict(size=18, color='gold', symbol='star',
                                line=dict(color='#8B0000', width=2)),
                    text=['★'], textposition='top center',
                    textfont=dict(size=16, color='#8B0000'),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>★ Optimal</b><br>"
                        f"PV {best['PV_kWp']:.0f} kWp | BESS {best['BESS_kWh']:.0f} kWh<br>"
                        f"LCOE ${best['LCOE_Total_$/kWh']:.4f}/kWh<extra></extra>"
                    ),
                    visible=(idx == 0),
                ))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='markers',
                                         visible=(idx == 0), showlegend=False))

        # build dropdown – each grid has 2 traces (heatmap + marker)
        n_grids = len(grid_limits)
        for idx, gl in enumerate(grid_limits):
            vis = [False] * (n_grids * 2)
            vis[idx * 2]     = True
            vis[idx * 2 + 1] = True
            buttons.append(dict(
                label=f'Grid {gl:.0f} kW',
                method='update',
                args=[{'visible': vis},
                      {'title': f'{label} — Grid {gl:.0f} kW'}],
            ))

        fig.update_layout(
            title=dict(text=f'{label} — Grid {grid_limits[0]:.0f} kW',
                       font=dict(size=17), x=0.5, xanchor='center'),
            xaxis=dict(title='PV Capacity (kWp)', showgrid=True,
                       gridcolor='#e0e0e0'),
            yaxis=dict(title='BESS Capacity (kWh)', showgrid=True,
                       gridcolor='#e0e0e0'),
            updatemenus=[dict(
                type='dropdown', direction='down',
                x=0.01, y=1.13, xanchor='left', yanchor='top',
                buttons=buttons,
                bgcolor='#ffffff', bordercolor='#cccccc', borderwidth=1,
                font=dict(size=13),
            )],
            height=520,
            margin=dict(l=80, r=130, t=80, b=70),
            paper_bgcolor='#ffffff', plot_bgcolor='#fafafa',
            font=dict(family='DM Sans, sans-serif', size=12, color='#2c3e50'),
        )
        return fig

    fig_renew   = _global_heatmap('Renewable_Fraction_%',     'Renewable Fraction (%)',        'YlGn',   False, '.1f')
    fig_curt    = _global_heatmap('Curtailment_Rate_%',       'PV Curtailment Rate (%)',       'RdYlGn', True,  '.1f')
    fig_cycles  = _global_heatmap('BESS_Cycles_per_Year',     'BESS Cycles / Year',            'Blues',  False, '.1f')
    fig_spill   = _global_heatmap('Generator_Spillage_Rate_%','Generator Spillage Rate (%)',   'OrRd',   False, '.1f')

    html_renew  = fig_renew .to_html(include_plotlyjs=False, div_id='plot_renew',  full_html=False)
    html_curt   = fig_curt  .to_html(include_plotlyjs=False, div_id='plot_curt',   full_html=False)
    html_cycles = fig_cycles.to_html(include_plotlyjs=False, div_id='plot_cycles', full_html=False)
    html_spill  = fig_spill .to_html(include_plotlyjs=False, div_id='plot_spill',  full_html=False)

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 5 – DISCOUNTED CUMULATIVE COST
    # ══════════════════════════════════════════════════════════════════════════
    html_cost = ""
    if optimal_sim_results and optimal_capex and discount_rate is not None:
        try:
            annual_results  = optimal_sim_results['annual_results']
            n_years         = len(annual_results)
            total_capex_val = sum(optimal_capex.values())
            base_pv_opex    = optimal_opex.get('pv_om', 0) if optimal_opex else 0
            base_bess_opex  = optimal_opex.get('bess_om', 0) if optimal_opex else 0
            base_diesel_om  = optimal_opex.get('diesel_om', 0) if optimal_opex else 0
            base_grid_om    = optimal_opex.get('grid_om', 0) if optimal_opex else 0
            base_om         = base_pv_opex + base_bess_opex + base_diesel_om + base_grid_om

            replacement_years_set = set(optimal_sim_results.get('replacement_years', []))
            bess_replacement_cost = sum([
                optimal_capex.get('bess_battery', 0),
                optimal_capex.get('bess_pcs', 0),
                optimal_capex.get('bess_bos', 0),
            ])

            years      = list(range(0, n_years + 1))
            cumulative = [0.0] * (n_years + 1)
            cumulative[0] = total_capex_val   # year 0 = CAPEX

            running = total_capex_val
            for yr in range(1, n_years + 1):
                r = annual_results[yr - 1]
                # variable backup costs
                var = 0.0
                if backup_type in ['Diesel', 'Grid+Diesel']:
                    var += r.get('diesel_cost_usd', 0) * (1 + diesel_escalation) ** (yr - 1)
                if backup_type in ['Grid', 'Grid+Diesel']:
                    var += r.get('grid_cost_usd', 0) * (1 + grid_escalation) ** (yr - 1)
                # replacement cost
                repl = bess_replacement_cost if yr in replacement_years_set else 0.0
                # total annual undiscounted cost
                annual_cost = base_om + var + repl
                # discount
                pv_cost     = annual_cost / ((1 + discount_rate) ** yr)
                running    += pv_cost
                cumulative[yr] = running

            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(
                x=years, y=[c / 1e6 for c in cumulative],
                mode='lines+markers',
                line=dict(color='#2563EB', width=3),
                marker=dict(size=6, color='#2563EB'),
                fill='tozeroy',
                fillcolor='rgba(37, 99, 235, 0.10)',
                name='Cumulative PV Cost',
                hovertemplate='Year %{x}<br>Cumulative Cost: $%{y:.2f}M<extra></extra>',
            ))
            # mark replacement years
            for ry in replacement_years_set:
                if 0 < ry <= n_years:
                    fig_cost.add_vline(
                        x=ry, line_dash='dash', line_color='#DC2626', line_width=1.5,
                        annotation_text=f'BESS replaced',
                        annotation_font=dict(color='#DC2626', size=11),
                        annotation_position='top right',
                    )

            fig_cost.update_layout(
                title=dict(text='Discounted Cumulative Project Cost (incl. CAPEX + PV of OPEX)',
                           font=dict(size=17), x=0.5, xanchor='center'),
                xaxis=dict(title='Year', tickmode='linear', dtick=2,
                           showgrid=True, gridcolor='#e5e7eb'),
                yaxis=dict(title='Cumulative Cost (M$)', showgrid=True,
                           gridcolor='#e5e7eb', tickformat='.2f'),
                height=420,
                margin=dict(l=80, r=40, t=70, b=60),
                paper_bgcolor='#ffffff', plot_bgcolor='#fafafa',
                font=dict(family='DM Sans, sans-serif', size=12, color='#2c3e50'),
                legend=dict(x=0.02, y=0.97),
            )
            html_cost = fig_cost.to_html(include_plotlyjs=False, div_id='plot_cost', full_html=False)
        except Exception as e:
            print(f"⚠ Could not build cost chart: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIG 6 – BATTERY CAPACITY OVER TIME
    # ══════════════════════════════════════════════════════════════════════════
    html_batt = ""
    if optimal_sim_results:
        try:
            annual_results = optimal_sim_results['annual_results']
            yrs  = [r['year'] + 1 for r in annual_results]        # year 1 … N
            caps = [r.get('battery_capacity_ratio', 1.0) * 100 for r in annual_results]
            # add year-0 starting point
            yrs  = [0] + yrs
            caps = [100.0] + caps

            repl_set = set(optimal_sim_results.get('replacement_years', []))

            fig_batt = go.Figure()
            fig_batt.add_trace(go.Scatter(
                x=yrs, y=caps,
                mode='lines+markers',
                line=dict(color='#059669', width=3),
                marker=dict(size=6, color='#059669'),
                fill='tozeroy',
                fillcolor='rgba(5, 150, 105, 0.10)',
                name='Battery Capacity',
                hovertemplate='Year %{x}<br>Capacity: %{y:.1f}%<extra></extra>',
            ))
            # replacement markers
            for ry in repl_set:
                if ry in yrs:
                    idx_ry = yrs.index(ry)
                    fig_batt.add_trace(go.Scatter(
                        x=[ry], y=[caps[idx_ry]],
                        mode='markers+text',
                        marker=dict(size=14, color='#DC2626', symbol='triangle-up'),
                        text=[' Replaced'], textposition='top right',
                        textfont=dict(color='#DC2626', size=11),
                        showlegend=False,
                        hovertemplate=f'Battery replaced at year {ry}<extra></extra>',
                    ))

            # reference lines at 80 % and 70 %
            for lvl, color in [(80, '#F59E0B'), (70, '#EF4444')]:
                fig_batt.add_hline(
                    y=lvl, line_dash='dot', line_color=color, line_width=1.5,
                    annotation_text=f'{lvl}% threshold',
                    annotation_font=dict(color=color, size=11),
                    annotation_position='right',
                )

            fig_batt.update_layout(
                title=dict(text='Battery Capacity Over Project Lifetime',
                           font=dict(size=17), x=0.5, xanchor='center'),
                xaxis=dict(title='Year', tickmode='linear', dtick=2,
                           showgrid=True, gridcolor='#e5e7eb', range=[-0.5, max(yrs) + 0.5]),
                yaxis=dict(title='Remaining Capacity (%)', showgrid=True,
                           gridcolor='#e5e7eb', range=[0, 105]),
                height=420,
                margin=dict(l=80, r=40, t=70, b=60),
                paper_bgcolor='#ffffff', plot_bgcolor='#fafafa',
                font=dict(family='DM Sans, sans-serif', size=12, color='#2c3e50'),
                legend=dict(x=0.02, y=0.05),
            )
            html_batt = fig_batt.to_html(include_plotlyjs=False, div_id='plot_batt', full_html=False)
        except Exception as e:
            print(f"⚠ Could not build battery chart: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # CAPEX / OPEX TABLES  (plain HTML)
    # ══════════════════════════════════════════════════════════════════════════
    def _make_table(data_dict, header_col2):
        if not data_dict:
            return "<p style='color:#888'>No data provided.</p>"
        rows = ""
        total = 0
        for k, v in data_dict.items():
            if v > 0:
                label = k.replace('_', ' ').replace('om', 'O&M').title()
                rows += f"<tr><td>{label}</td><td>${v:,.0f}</td></tr>\n"
                total += v
        rows += (f"<tr class='total-row'>"
                 f"<td><strong>Total</strong></td>"
                 f"<td><strong>${total:,.0f}</strong></td></tr>\n")
        return f"""
        <table class='fin-table'>
          <thead><tr><th>Component</th><th>{header_col2}</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    capex_table = _make_table(optimal_capex or {}, 'Cost ($)')
    opex_table  = _make_table(optimal_opex  or {}, 'Annual Cost ($)')

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMAL CONFIG METRICS
    # ══════════════════════════════════════════════════════════════════════════
    opt_pv        = best['PV_kWp']
    opt_bess      = best['BESS_kWh']
    opt_grid      = best['Grid_Limit_kW']
    opt_lcoe      = best['LCOE_Total_$/kWh']
    opt_renew     = best['Renewable_Fraction_%']
    opt_curt      = best.get('Curtailment_Rate_%', 0)
    opt_cycles    = best.get('BESS_Cycles_per_Year', 0)
    opt_capex_val = best.get('Total_CAPEX_$', sum(optimal_capex.values()) if optimal_capex else 0)
    opt_batt_repl = best.get('Battery_Replacements', 0)
    opt_spillage  = best.get('Generator_Spillage_MWh', 0)

    # ══════════════════════════════════════════════════════════════════════════
    # LOAD SECTION  (only if data provided)
    # NOTE: this is the FIRST plot in the HTML → must carry include_plotlyjs='cdn'
    # All subsequent plots use include_plotlyjs=False
    # ══════════════════════════════════════════════════════════════════════════
    load_html_snippet = ""
    if load_hourly_data is not None and isinstance(load_hourly_data, pd.DataFrame):
        col = next((c for c in ['Load_kW', 'load_kw', 'load'] if c in load_hourly_data.columns), None)
        if col:
            raw = load_hourly_data[col].values.astype(float)
            # Guarantee exactly 8760 points
            if len(raw) >= 8760:
                arr = raw[:8760]
            else:
                arr = np.pad(raw, (0, 8760 - len(raw)), constant_values=0.0)
            try:
                # reshape: 365 days × 24 hours, then transpose → (24, 365)
                # rows = hours 0-23, cols = days 1-365
                mat = arr.reshape(365, 24).T

                fig_load = go.Figure(go.Heatmap(
                    z=mat,
                    x=list(range(1, 366)),   # day of year
                    y=list(range(0, 24)),     # hour of day
                    colorscale='RdYlGn',
                    reversescale=True ,
                    colorbar=dict(
                        title=dict(text='kW', side='right'),
                        thickness=18, len=0.85, x=1.01,
                    ),
                    hovertemplate='Day %{x}<br>Hour %{y}:00<br>Load: %{z:.1f} kW<extra></extra>',
                ))
                fig_load.update_layout(
                    title=dict(
                        text='Annual Hourly Load Pattern  (Hour of Day × Day of Year)',
                        font=dict(size=16), x=0.5, xanchor='center',
                    ),
                    xaxis=dict(title='Day of Year', showgrid=False, zeroline=False),
                    yaxis=dict(
                        title='Hour of Day', showgrid=False, zeroline=False,
                        tickmode='array',
                        tickvals=list(range(0, 24, 3)),
                        ticktext=[f'{h:02d}:00' for h in range(0, 24, 3)],
                        autorange='reversed',   # hour 0 at top, 23 at bottom
                    ),
                    height=400,
                    margin=dict(l=70, r=110, t=55, b=55),
                    paper_bgcolor='#ffffff',
                    plot_bgcolor='#ffffff',
                    font=dict(family='DM Sans, sans-serif', size=12, color='#2c3e50'),
                )
                # ← 'cdn' here because this is the FIRST plotly chart in the HTML
                load_html_snippet = fig_load.to_html(
                    include_plotlyjs='cdn', div_id='plot_load', full_html=False)
                print("✓ Load heatmap created")
            except Exception as e:
                print(f"⚠ Load heatmap failed: {e}")

    loc_str = (f"{latitude:.4f}° N, {longitude:.4f}° E"
               if latitude is not None and longitude is not None else "N/A")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 STAT CARDS
    # ══════════════════════════════════════════════════════════════════════════
    def _card(label, value):
        return f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}</div>
        </div>"""

    lcoe_min = df['LCOE_Total_$/kWh'].min()
    lcoe_max = df['LCOE_Total_$/kWh'].max()
    pv_range  = f"{df['PV_kWp'].min():.0f} – {df['PV_kWp'].max():.0f} kWp"
    bess_range= f"{df['BESS_kWh'].min():.0f} – {df['BESS_kWh'].max():.0f} kWh"
    grid_range= f"{df['Grid_Limit_kW'].min():.0f} – {df['Grid_Limit_kW'].max():.0f} kW"

    stat_cards = (
        _card("Total Configurations", f"{len(df)}")
      + _card("PV Range", pv_range)
      + _card("BESS Range", bess_range)
      + _card("Grid Range", grid_range)
      + _card("LCOE Range", f"${lcoe_min:.3f} – ${lcoe_max:.3f}/kWh")
      + _card("Discount Rate", dr_pct)
    )

    # ══════════════════════════════════════════════════════════════════════════
    # COST & BATTERY CHART SECTIONS  (only if charts built)
    # ══════════════════════════════════════════════════════════════════════════
    cost_section = f"""
    <div class="section">
      <h2>Discounted Cumulative Project Cost</h2>
      <p class="section-desc">
        Cumulative present-value of all project expenditures (CAPEX at year 0 plus
        discounted OPEX and any battery replacement costs). Dashed red lines mark
        BESS replacement events.
      </p>
      <div class="plot-box">{html_cost}</div>
    </div>""" if html_cost else ""

    batt_section = f"""
    <div class="section">
      <h2>Battery Capacity Over Project Lifetime</h2>
      <p class="section-desc">
        Simulated remaining capacity of the battery system year-by-year.
        Calendar and cycle ageing are modelled; red triangles mark replacement events.
        Dashed horizontal lines indicate the 80 % and 70 % replacement thresholds.
      </p>
      <div class="plot-box">{html_batt}</div>
    </div>""" if html_batt else ""

    # load_html_snippet is the raw plotly div; it goes straight into the HTML body
    # (the body already wraps it in a plot-box, so no extra wrapper here)

    # ══════════════════════════════════════════════════════════════════════════
    # ASSEMBLE HTML
    # ══════════════════════════════════════════════════════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{project_name}</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'DM Sans', sans-serif;
      background: #f0f2f5;
      color: #1e293b;
      line-height: 1.6;
    }}

    /* ── TOP HEADER ── */
    .header {{
      background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
      color: #fff;
      padding: 48px 60px 40px;
      border-bottom: 4px solid #3b82f6;
    }}
    .header h1 {{
      font-size: 2.2rem;
      font-weight: 700;
      letter-spacing: -0.5px;
      margin-bottom: 8px;
    }}
    .header p {{
      font-size: 1rem;
      opacity: 0.75;
      font-weight: 300;
    }}
    .header .ts {{
      font-family: 'DM Mono', monospace;
      font-size: 0.78rem;
      opacity: 0.55;
      margin-top: 10px;
    }}

    /* ── MAIN WRAPPER ── */
    .wrapper {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 40px 40px 80px;
    }}

    /* ── SECTION ── */
    .section {{
      margin-bottom: 56px;
    }}
    .section h2 {{
      font-size: 1.45rem;
      font-weight: 700;
      color: #0f172a;
      border-left: 5px solid #3b82f6;
      padding-left: 14px;
      margin-bottom: 10px;
    }}
    .section-desc {{
      color: #64748b;
      font-size: 0.92rem;
      margin-bottom: 18px;
      max-width: 820px;
    }}

    /* ── ROW 1  (location + load) ── */
    .row-top {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
      margin-bottom: 40px;
    }}
    .info-panel {{
      background: #fff;
      border-radius: 12px;
      padding: 28px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
      border: 1px solid #e2e8f0;
    }}
    .info-panel h3 {{
      font-size: 1rem;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 16px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      font-size: 0.78rem;
    }}
    .info-row {{
      display: flex;
      justify-content: space-between;
      padding: 7px 0;
      border-bottom: 1px solid #f1f5f9;
      font-size: 0.9rem;
    }}
    .info-row:last-child {{ border-bottom: none; }}
    .info-row span:first-child {{ color: #64748b; }}
    .info-row span:last-child  {{ font-weight: 600; color: #0f172a; }}

    /* ── STAT CARDS (row 2) ── */
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 18px;
      margin-bottom: 48px;
    }}
    .stat-card {{
      background: #fff;
      border-radius: 10px;
      padding: 22px 20px;
      text-align: center;
      box-shadow: 0 1px 8px rgba(0,0,0,0.07);
      border: 1px solid #e2e8f0;
      border-top: 4px solid #3b82f6;
      transition: transform 0.15s;
    }}
    .stat-card:hover {{ transform: translateY(-3px); }}
    .stat-label {{
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      color: #64748b;
      margin-bottom: 8px;
    }}
    .stat-value {{
      font-size: 1.35rem;
      font-weight: 700;
      color: #0f172a;
    }}

    /* ── PLOT BOX ── */
    .plot-box {{
      background: #fff;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
      border: 1px solid #e2e8f0;
      margin-bottom: 24px;
      overflow: hidden;
    }}

    /* ── OPTIMAL CONFIG CARD ── */
    .optimal-card {{
      background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
      border-radius: 14px;
      padding: 34px 36px;
      color: #fff;
      box-shadow: 0 4px 20px rgba(6, 78, 59, 0.35);
      margin-bottom: 48px;
    }}
    .optimal-card h2 {{
      font-size: 1.4rem;
      font-weight: 700;
      margin-bottom: 24px;
      border-left: 5px solid #6ee7b7;
      padding-left: 14px;
      color: #fff;
    }}
    .opt-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
      margin-bottom: 30px;
    }}
    .opt-metric {{
      background: rgba(255,255,255,0.12);
      border-radius: 8px;
      padding: 16px 18px;
      text-align: center;
      backdrop-filter: blur(6px);
    }}
    .opt-metric .lbl {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      opacity: 0.75;
      margin-bottom: 6px;
    }}
    .opt-metric .val {{
      font-size: 1.55rem;
      font-weight: 700;
    }}
    .opt-metric .val.big {{ font-size: 1.85rem; }}

    /* ── FINANCIAL TABLES ── */
    .fin-tables {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 10px;
    }}
    .fin-table-wrap {{
      background: rgba(255,255,255,0.12);
      border-radius: 10px;
      padding: 20px;
    }}
    .fin-table-wrap h4 {{
      font-size: 0.9rem;
      font-weight: 700;
      margin-bottom: 14px;
      opacity: 0.9;
      border-bottom: 1px solid rgba(255,255,255,0.25);
      padding-bottom: 8px;
    }}
    .fin-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    .fin-table th {{
      text-align: left;
      padding: 6px 8px;
      font-weight: 600;
      opacity: 0.7;
      border-bottom: 1px solid rgba(255,255,255,0.2);
    }}
    .fin-table td {{
      padding: 6px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .fin-table .total-row td {{
      border-top: 2px solid rgba(255,255,255,0.35);
      border-bottom: none;
      padding-top: 10px;
      font-weight: 600;
    }}

    /* ── FOOTER ── */
    .footer {{
      text-align: center;
      padding: 30px;
      color: #94a3b8;
      font-size: 0.82rem;
    }}

    @media (max-width: 900px) {{
      .row-top     {{ grid-template-columns: 1fr; }}
      .fin-tables  {{ grid-template-columns: 1fr; }}
      .wrapper     {{ padding: 20px; }}
    }}
  </style>
</head>
<body>

<div class="header">
  <h1>⚡ {project_name}</h1>
  <p>Prefeasibility Analysis — Hybrid PV + BESS LCOE Optimisation</p>
  <p class="ts">Generated on {timestamp}</p>
</div>

<div class="wrapper">

  <!-- ══════════════════════════════════════════════════════════════
       ROW 1  Location info  +  Load heatmap
  ══════════════════════════════════════════════════════════════ -->
  <div class="row-top" style="margin-top:36px">

    <div class="info-panel">
      <h3>Project Description</h3>
      <div class="info-row">
        <span>Location</span>
        <span>{loc_str}</span>
      </div>
      <div class="info-row">
        <span>Backup Type</span>
        <span>{backup_type or 'N/A'}</span>
      </div>
      <div class="info-row">
        <span>Annual Load</span>
        <span>{load_annual_mwh:,.1f} MWh</span>
      </div>
      <div class="info-row">
        <span>Avg Load</span>
        <span>{load_avg_kw:,.1f} kW</span>
      </div>
      <div class="info-row">
        <span>Peak Load</span>
        <span>{load_peak_kw:,.1f} kW</span>
      </div>
      <div class="info-row">
        <span>Project Life</span>
        <span>{len(optimal_sim_results["annual_results"]) if optimal_sim_results else "N/A"} years</span>
      </div>
      <div class="info-row">
        <span>Discount Rate</span>
        <span>{dr_pct}</span>
      </div>
    </div>

    <div class="plot-box" style="margin-bottom:0">
      <h3 style="font-size:0.95rem;font-weight:600;color:#0f172a;margin-bottom:12px">
        Annual Load Profile Heatmap
      </h3>
      {load_html_snippet if load_html_snippet else
       '<p style="color:#94a3b8;padding:60px;text-align:center">Load heatmap not available — no load data provided</p>'}
    </div>

  </div>

  <!-- ══════════════════════════════════════════════════════════════
       ROW 2  Summary stat cards
  ══════════════════════════════════════════════════════════════ -->
  <div class="stats-row">
    {stat_cards}
  </div>

  <!-- ══════════════════════════════════════════════════════════════
       3-D SCATTER
  ══════════════════════════════════════════════════════════════ -->
  <div class="section">
    <h2>3-D Configuration Space — PV × BESS × LCOE</h2>
    <p class="section-desc">
      Each point is one simulated configuration. Colour maps LCOE:
      green = low cost, red = high cost. The gold diamond marks the optimal (minimum LCOE) configuration.
      Drag to rotate; scroll to zoom.
    </p>
    <div class="plot-box">{html_3d}</div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════
       2-D HEATMAPS
  ══════════════════════════════════════════════════════════════ -->
  <div class="section">
    <h2>2-D Heatmaps by Grid Limit</h2>
    <p class="section-desc">
      Use the dropdown to switch between grid limit scenarios.
      The colour scale is <strong>fixed globally</strong> across all grid limits so comparisons
      between scenarios are visually consistent. ★ marks the overall optimal configuration.
    </p>

    <div class="plot-box">
      <h3 style="font-size:1rem;font-weight:600;color:#0f172a;margin-bottom:4px">
        1 — Renewable Energy Fraction  (higher = better)
      </h3>
      <p style="color:#64748b;font-size:0.88rem;margin-bottom:12px">
        Percentage of served load met by PV + BESS. Deeper green = more renewable supply.
      </p>
      {html_renew}
    </div>

    <div class="plot-box">
      <h3 style="font-size:1rem;font-weight:600;color:#0f172a;margin-bottom:4px">
        2 — Energy Curtailment Rate  (lower = better)
      </h3>
      <p style="color:#64748b;font-size:0.88rem;margin-bottom:12px">
        Fraction of generated PV energy that must be curtailed due to excess.
        Red = high curtailment (oversized PV with insufficient BESS).
      </p>
      {html_curt}
    </div>

    <div class="plot-box">
      <h3 style="font-size:1rem;font-weight:600;color:#0f172a;margin-bottom:4px">
        3 — BESS Cycles per Year
      </h3>
      <p style="color:#64748b;font-size:0.88rem;margin-bottom:12px">
        Average full charge-discharge cycles per year. Affects battery lifetime and replacement frequency.
        Deeper blue = more cycling (higher utilisation).
      </p>
      {html_cycles}
    </div>

    <div class="plot-box">
      <h3 style="font-size:1rem;font-weight:600;color:#0f172a;margin-bottom:4px">
        4 — Generator Spillage Rate  (lower = better)
      </h3>
      <p style="color:#64748b;font-size:0.88rem;margin-bottom:12px">
        Diesel generator minimum-load excess that cannot be absorbed by the battery, expressed as a
        percentage of total diesel generation. Only relevant when backup type includes diesel.
        Deeper orange-red = more wasted generator output.
      </p>
      {html_spill}
    </div>

    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════
       OPTIMAL CONFIGURATION CARD
  ══════════════════════════════════════════════════════════════ -->
  <div class="optimal-card">
    <h2>★ Optimal Configuration (Minimum LCOE)</h2>

    <div class="opt-grid">
      <div class="opt-metric">
        <div class="lbl">PV Capacity</div>
        <div class="val">{opt_pv:,.0f} <small>kWp</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">BESS Capacity</div>
        <div class="val">{opt_bess:,.0f} <small>kWh</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">Grid Capacity</div>
        <div class="val">{opt_grid:,.0f} <small>kW</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">LCOE</div>
        <div class="val big">${opt_lcoe:.4f} <small>/kWh</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">Renewable Fraction</div>
        <div class="val">{opt_renew:.1f}<small>%</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">PV Curtailment</div>
        <div class="val">{opt_curt:.1f}<small>%</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">Generator Spillage</div>
        <div class="val">{opt_spillage:.1f}<small> MWh/yr</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">BESS Cycles/yr</div>
        <div class="val">{opt_cycles:.0f}</div>
      </div>
      <div class="opt-metric">
        <div class="lbl">Total CAPEX</div>
        <div class="val">${opt_capex_val/1e6:.2f}<small>M</small></div>
      </div>
      <div class="opt-metric">
        <div class="lbl">Battery Replacements</div>
        <div class="val">{int(opt_batt_repl)}</div>
      </div>
    </div>

    <div class="fin-tables">
      <div class="fin-table-wrap">
        <h4>CAPEX Breakdown</h4>
        {capex_table}
      </div>
      <div class="fin-table-wrap">
        <h4>Annual OPEX Breakdown</h4>
        {opex_table}
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════════
       COST CURVE
  ══════════════════════════════════════════════════════════════ -->
  {cost_section}

  <!-- ══════════════════════════════════════════════════════════════
       BATTERY AGEING
  ══════════════════════════════════════════════════════════════ -->
  {batt_section}

</div><!-- /wrapper -->

<div class="footer">
  {project_name} &nbsp;|&nbsp; {len(df)} configurations analysed &nbsp;|&nbsp; Generated {timestamp}
</div>

</body>
</html>"""

    print("✓ HTML report generated successfully.")
    return html.encode('utf-8')
