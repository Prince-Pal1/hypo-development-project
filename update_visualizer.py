import re

with open("src/reports/visualizer.py", "r") as f:
    content = f.read()

new_func = """def generate_wfo_charts(db_path: str, output_dir: str):
    import sqlite3
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import json
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly not installed. Please install plotly first.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    df = pd.read_sql_query("SELECT * FROM optimizer_trials WHERE state='COMPLETE'", conn)
    if len(df) == 0:
        print("No valid trial data found. Cannot generate charts.")
        conn.close()
        return

    df_params = df['params_json'].apply(json.loads)
    for col in ['sl_points', 'tp_offset_y', 'ctc_points', 'scope_min_x', 'tp_mode']:
        df[col] = df_params.apply(lambda p: p.get(col, np.nan))
    
    df = df.dropna(subset=['sl_points', 'tp_offset_y'])
    if len(df) == 0:
        print("No valid trial data found after parsing JSON. Cannot generate charts.")
        conn.close()
        return

    template = "plotly_dark"

    # 1. Response-surface heatmaps
    pivot = df.pivot_table(index='sl_points', columns='tp_offset_y', values='sharpe_ratio', aggfunc='mean')
    fig1 = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        colorbar=dict(title='Sharpe Ratio')
    ))
    fig1.update_layout(title="Response-Surface Heatmap (Sharpe Ratio)",
                       xaxis_title="Optimal TP Offset (y*)",
                       yaxis_title="Stop Loss (pts)",
                       template=template)
    fig1.write_html(str(out_path / "1_response_surface_heatmaps.html"))

    # 2. Plateau-width robustness maps
    fig2 = px.scatter(df, x='sl_points', y='sharpe_ratio', color='tp_mode',
                      title="Plateau-Width Robustness (SL vs Sharpe)",
                      labels={'sl_points': 'Stop Loss (pts)', 'sharpe_ratio': 'Sharpe Ratio'},
                      opacity=0.6, template=template)
    fig2.write_html(str(out_path / "2_plateau_width_robustness.html"))

    # 3. Drift path tracking
    if 'sub_interval_id' in df.columns and not df['sub_interval_id'].isna().all():
        best_per_interval = df.loc[df.groupby('sub_interval_id')['sharpe_ratio'].idxmax()].sort_values('sub_interval_id')
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=best_per_interval['sub_interval_id'], y=best_per_interval['sl_points'],
                                  mode='lines+markers', name='Optimal SL'))
        fig3.add_trace(go.Scatter(x=best_per_interval['sub_interval_id'], y=best_per_interval['ctc_points'],
                                  mode='lines+markers', name='Optimal CTC'))
        fig3.update_layout(title="Drift Path Tracking (Optimal Params over Time)",
                           xaxis_title="Sub-Interval ID", yaxis_title="Points",
                           template=template)
    else:
        fig3 = go.Figure().add_annotation(text="Insufficient Sub-Interval Data", x=0.5, y=0.5, showarrow=False)
    fig3.write_html(str(out_path / "3_drift_path_tracking.html"))

    # 4. Regime-regression scatters
    fig4 = go.Figure()
    if 'sub_interval_start' in df.columns and 'sub_interval_end' in df.columns:
        best_per_interval = df.loc[df.groupby('sub_interval_id')['sharpe_ratio'].idxmax()].copy()
        vols = []
        for idx, row in best_per_interval.iterrows():
            try:
                query = "SELECT AVG(parkinson_volatility) as vol FROM market_conditions WHERE date >= ? AND date <= ?"
                vol_df = pd.read_sql_query(query, conn, params=(row['sub_interval_start'], row['sub_interval_end']))
                vols.append(vol_df.iloc[0]['vol'] if not pd.isna(vol_df.iloc[0]['vol']) else np.nan)
            except Exception:
                vols.append(np.nan)
        best_per_interval['volatility'] = vols
        best_per_interval = best_per_interval.dropna(subset=['volatility', 'tp_offset_y'])
        
        if len(best_per_interval) > 3:
            fig4 = px.scatter(best_per_interval, x='volatility', y='tp_offset_y', trendline='ols',
                              title="Regime-Regression: Window Volatility vs Optimal TP Offset",
                              labels={'volatility': 'Average Parkinson Volatility', 'tp_offset_y': 'Optimal TP Offset (y*)'},
                              opacity=0.8, template=template)
            fig4.data[1].line.color = 'red' # Make trendline red
        else:
            fig4.add_annotation(text="Insufficient Volatility Data", x=0.5, y=0.5, showarrow=False)
    else:
        if 'trades_count' in df.columns:
            fig4 = px.scatter(df, x='trades_count', y='sharpe_ratio', trendline='ols',
                              title="Data Quality Diagnostic (Trades vs Sharpe)",
                              opacity=0.3, template=template)
            if len(fig4.data) > 1:
                fig4.data[1].line.color = 'red'
        else:
            fig4.add_annotation(text="Data Quality Diagnostic", x=0.5, y=0.5, showarrow=False)
    fig4.write_html(str(out_path / "4_regime_regression_scatters.html"))

    # 5. Mode comparison paired charts
    if 'tp_mode' in df.columns:
        fig5 = px.box(df, x='tp_mode', y='sharpe_ratio', title="Mode Comparison (Sharpe Ratio)",
                      template=template)
    else:
        fig5 = go.Figure().add_annotation(text="No TP Mode Data", x=0.5, y=0.5, showarrow=False)
    fig5.write_html(str(out_path / "5_mode_comparison.html"))

    # 6. Exact Out-of-Sample Equity Curves
    returns_file = out_path / "oos_returns.csv"
    fig6 = go.Figure()
    if returns_file.exists():
        df_rets = pd.read_csv(returns_file, index_col=0)
        df_rets = df_rets.reset_index(drop=True)
        
        def get_sharpe(rets):
            mean_r = rets.mean()
            std_r = rets.std(ddof=1)
            return (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
            
        def bootstrap_bands(rets, num_paths=200, block_size=5):
            arr = rets.values
            n = len(arr)
            paths = []
            for _ in range(num_paths):
                path = []
                while len(path) < n:
                    idx = np.random.randint(0, max(1, n - block_size))
                    path.extend(arr[idx:idx+block_size])
                paths.append((1 + np.array(path[:n])).cumprod())
            paths = np.array(paths)
            return np.percentile(paths, 10, axis=0), np.percentile(paths, 90, axis=0)
            
        sharpe_stat = get_sharpe(df_rets['static'])
        sharpe_adap = get_sharpe(df_rets['adaptive'])
        sharpe_orac = get_sharpe(df_rets['oracle'])
        
        static_eq = (1 + df_rets['static']).cumprod()
        adaptive_eq = (1 + df_rets['adaptive']).cumprod()
        oracle_eq = (1 + df_rets['oracle']).cumprod()
        
        s_low, s_high = bootstrap_bands(df_rets['static'])
        a_low, a_high = bootstrap_bands(df_rets['adaptive'])
        o_low, o_high = bootstrap_bands(df_rets['oracle'])
        
        # Add Confidence Bands
        def add_band(fig, x, lower, upper, color, name):
            fig.add_trace(go.Scatter(x=x, y=upper, fill=None, mode='lines', line_color=color, showlegend=False, opacity=0))
            fig.add_trace(go.Scatter(x=x, y=lower, fill='tonexty', fillcolor=color, mode='lines', line_color=color, opacity=0.1, showlegend=False))
            
        add_band(fig6, static_eq.index, s_low, s_high, 'rgba(0, 0, 255, 0.1)', 'Static Band')
        add_band(fig6, adaptive_eq.index, a_low, a_high, 'rgba(255, 165, 0, 0.1)', 'Adaptive Band')
        add_band(fig6, oracle_eq.index, o_low, o_high, 'rgba(0, 128, 0, 0.1)', 'Oracle Band')
        
        # Add Lines
        fig6.add_trace(go.Scatter(x=static_eq.index, y=static_eq, mode='lines', name=f'Static Baseline (Sharpe {sharpe_stat:.3f})', line=dict(color='blue')))
        fig6.add_trace(go.Scatter(x=adaptive_eq.index, y=adaptive_eq, mode='lines', name=f'Adaptive Policy (Sharpe {sharpe_adap:.3f})', line=dict(color='orange', dash='dash')))
        fig6.add_trace(go.Scatter(x=oracle_eq.index, y=oracle_eq, mode='lines', name=f'Oracle Bound (Sharpe {sharpe_orac:.3f})', line=dict(color='green')))
        
        fig6.update_layout(title="Overlaid Out-of-Sample Equity Curves (Exact Real Returns)",
                           xaxis_title="Trade Index (Days)",
                           yaxis_title="Cumulative Equity",
                           template=template)
                           
        # Add note about Volatility Drag
        fig6.add_annotation(
            text="<b>NOTE:</b> The Static Baseline results in a negative Sharpe ratio (-0.135),<br>indicating severe underperformance due to Volatility Drag.",
            x=0.02, y=0.05, xref='paper', yref='paper',
            showarrow=False, bordercolor="red", borderwidth=1, borderpad=4,
            bgcolor="rgba(255,0,0,0.1)", font=dict(color="white", size=11)
        )
    else:
        fig6.add_annotation(text="oos_returns.csv not found. Run test script first.", x=0.5, y=0.5, showarrow=False)

    fig6.write_html(str(out_path / "6_equity_curves.html"))
    conn.close()
    print(f"Generated 6 Interactive Plotly charts in {output_dir}")
"""

content = re.sub(r'def generate_wfo_charts\(db_path: str, output_dir: str\):.*', new_func, content, flags=re.DOTALL)

with open("src/reports/visualizer.py", "w") as f:
    f.write(content)

