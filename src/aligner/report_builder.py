import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from aligner.plot_utils import *

class HTMLReportBuilder:
    def __init__(self, growth_df=None):
        self.growth_df = growth_df

    def _get_plotly_optimization_landscape(self, slice_landscape):
        coarse_history = slice_landscape.get('coarse', [])
        tournament = slice_landscape.get('tournament', [])
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Coarse Roll Landscape", "ICP Convergence"))
        
        df_coarse = pd.DataFrame(coarse_history)
        if not df_coarse.empty:
            df_plus = df_coarse[df_coarse['sign'] == 1.0].sort_values('angle')
            df_minus = df_coarse[df_coarse['sign'] == -1.0].sort_values('angle')
            fig.add_trace(go.Scatter(x=df_plus['angle'].tolist(), y=df_plus['cost'].tolist(), name='+PC1 Axis', line=dict(color='#3498db')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_minus['angle'].tolist(), y=df_minus['cost'].tolist(), name='-PC1 Axis', line=dict(color='#e74c3c')), row=1, col=1)

        for finalist in tournament:
            angle = finalist['init_angle']
            start_point = df_coarse[np.isclose(df_coarse['angle'], angle, atol=1e-5)]
            if not start_point.empty:
                cost = start_point['cost'].min()
                fig.add_trace(go.Scatter(
                    x=[angle], y=[cost], mode='markers+text',
                    marker=dict(color='#f1c40f', size=12, symbol='star', line=dict(color='black', width=1)),
                    text=[f"R{finalist['start_rank']}"], textposition="top center", showlegend=False
                ), row=1, col=1)

        colors = ['#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#34495e']
        for i, finalist in enumerate(tournament):
            icp_trace = finalist.get('icp_history', [])
            if icp_trace:
                df_icp = pd.DataFrame(icp_trace)
                fig.add_trace(go.Scatter(
                    x=df_icp['iter'].tolist(), y=df_icp['cost'].tolist(), mode='lines+markers',
                    name=f"R{finalist['start_rank']} (N={len(finalist.get('labels', []))})", 
                    line=dict(color=colors[i % len(colors)], width=2)
                ), row=1, col=2)

        # --- NEW: Explicit Axis Labels ---
        fig.update_xaxes(title_text="Roll Angle (°)", row=1, col=1)
        fig.update_yaxes(title_text="Alignment Cost", row=1, col=1)
        fig.update_xaxes(title_text="ICP Iteration", row=1, col=2)
        fig.update_yaxes(title_text="Alignment Cost", row=1, col=2)

        fig.update_layout(autosize=True, height=350, plot_bgcolor='white', margin=dict(t=30, b=40, l=40, r=30), showlegend=True)
        return fig

    def _generate_global_summary_table(self, landscape_dict, winner_slice_id, winner_rank):
        all_valleys = []
        for s_id, s_data in landscape_dict.items():
            for finalist in s_data['tournament']:
                entry = finalist.copy()
                entry['hypothesis_id'] = s_id
                entry['source'] = "Aug" if finalist.get('is_generated', False) else "Obs"
                all_valleys.append(entry)
        
        all_valleys = sorted(all_valleys, key=lambda x: x['cost'])
        
        html = """<table class="summary-table"><thead><tr>
            <th>Slice</th><th>Type</th><th>Rank</th><th>Cost</th><th>Conf</th><th>Slack</th>
            </tr></thead><tbody>"""
        for v in all_valleys:
            is_global_winner = "winner-row" if (v['hypothesis_id'] == winner_slice_id and v['start_rank'] == winner_rank) else ""
            html += f"""
            <tr class="{is_global_winner}">
                <td><b>{v['hypothesis_id']}</b></td>
                <td><span class="tiny-tag {'tag-gen' if v['source']=='Aug' else 'tag-obs'}">{v['source']}</span></td>
                <td>R{v['start_rank']}</td>
                <td>{v['cost']:.1f}</td>
                <td>{v.get('mean_confidence', 0.0):.1%}</td>
                <td>{v.get('slack_count', 0)}</td>
            </tr>"""
        html += "</tbody></table>"
        return html

    def build_report(self, report_package, output_path="alignment_explorer.html"):
        best_res = report_package['best_result']
        landscape = report_package['landscape']
        
        global_summary_table = self._generate_global_summary_table(landscape, best_res['slice_id'], best_res['start_rank'])

        if self.growth_df is not None:
            fig_temp = get_plotly_temporal_context(self.growth_df, len(best_res['labels']), best_res['map_time'])
            div_temp = fig_temp.to_html(full_html=False, include_plotlyjs='cdn')
        else:
            empty_fig = go.Figure()
            div_temp = "<p>Growth data missing.</p>" + empty_fig.to_html(full_html=False, include_plotlyjs='cdn')

        # --- Extract Final Stats for Header ---
        winning_slice_id = best_res['slice_id']
        winning_rank = best_res['start_rank']
        
        winning_outcome = next((v for v in landscape[winning_slice_id]['tournament'] if v['start_rank'] == winning_rank), best_res)
        final_conf = winning_outcome.get('mean_confidence', 0.0)
        final_n_cells = len(winning_outcome.get('labels', []))

        plot_data_store = {}
        slice_ids = sorted(list(landscape.keys()))
        tab_buttons = ""
        slice_sections = ""

        for s_id in slice_ids:
            tournament = landscape[s_id]['tournament']
            is_top_winner = (s_id == winning_slice_id)
            n_cells = len(tournament[0].get('labels', []))
            
            plot_data_store[str(s_id)] = {}
            
            active_tab = "active" if is_top_winner else ""
            win_label = " (Winner)" if is_top_winner else ""
            tab_buttons += f'<button class="tablinks {active_tab}" onclick="openSlice(event, \'{s_id}\')">Slice {s_id}{win_label}</button>'
            
            div_opt = self._get_plotly_optimization_landscape(landscape[s_id]).to_html(full_html=False, include_plotlyjs=False)

            minima_dropdown = ""
            for outcome in tournament:
                rank = str(outcome['start_rank'])
                minima_dropdown += f'<option value="{rank}">Rank {rank} (Cost: {outcome["cost"]:.1f})</option>'
                
                fig_align = plot_inference_alignment_interactive(outcome)
                fig_conf = plot_spatial_confidence_interactive(outcome)
                
                plot_data_store[str(s_id)][rank] = {
                    'align': json.loads(fig_align.to_json()),
                    'conf': json.loads(fig_conf.to_json())
                }

            slice_sections += f"""
            <div id="slice-{s_id}" class="tabcontent" style="display: {"block" if is_top_winner else "none"};">
                <div class="slice-meta">Hypothesis ID: <b>{s_id}</b> | Cells: <b>{n_cells}</b></div>
                <div class="card full-width"><h4>Search Landscape & Convergence</h4>{div_opt}</div>
                <div class="selector-bar">
                    <label><b>View Local Minimum:</b> </label>
                    <select id="select-{s_id}" onchange="renderPlots('{s_id}', this.value)">{minima_dropdown}</select>
                </div>
                <div class="top-grid">
                    <div class="card"><h4>3D Alignment View</h4><div id="align-container-{s_id}" style="height: 500px;"></div></div>
                    <div class="card"><h4>Spatial Confidence Map</h4><div id="conf-container-{s_id}" style="height: 500px;"></div></div>
                </div>
            </div>
            """

        js_data_dump = json.dumps(plot_data_store)

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Alignment Explorer | {report_package['embryo_id']}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; margin: 0; padding: 20px; }}
                
                .main-header {{ background: #2c3e50; color: white; padding: 20px 30px; border-radius: 8px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .main-header h1 {{ margin: 0; font-size: 26px; font-weight: 600; letter-spacing: 0.5px; }}
                .main-header .stats {{ display: flex; gap: 40px; }}
                .main-header .stat-box {{ display: flex; flex-direction: column; align-items: flex-end; }}
                .main-header .stat-value {{ font-size: 24px; font-weight: bold; color: #2ecc71; }}
                .main-header .stat-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.8; margin-top: 2px; }}
                
                .top-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
                .card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
                .full-width {{ width: 100%; margin-bottom: 20px; box-sizing: border-box; }}
                .summary-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
                .summary-table th, .summary-table td {{ padding: 6px; border: 1px solid #eee; text-align: center; }}
                .winner-row {{ background: #f0f7ff; border: 2px solid #007bff !important; }}
                .tiny-tag {{ padding: 2px 5px; border-radius: 4px; color: white; font-size: 10px; font-weight: bold; }}
                .tag-obs {{ background: #27ae60; }} .tag-gen {{ background: #8e44ad; }}
                .tab {{ overflow: hidden; border-bottom: 2px solid #ddd; }}
                .tab button {{ background: #e0e0e0; border: none; cursor: pointer; padding: 10px 20px; border-radius: 8px 8px 0 0; margin-right: 4px; }}
                .tab button.active {{ background: white; border: 1px solid #ddd; border-bottom: 2px solid white; color: #007bff; font-weight: bold; }}
                .tabcontent {{ background: white; padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 8px 8px; }}
                .selector-bar {{ background: #e9ecef; padding: 12px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #007bff; }}
                h4 {{ margin-top: 0; color: #34495e; border-bottom: 2px solid #f1f1f1; padding-bottom: 8px; }}
            </style>
        </head>
        <body>
            <div class="main-header">
                <h1>Embryo: {report_package['embryo_id']}</h1>
                <div class="stats">
                    <div class="stat-box">
                        <span class="stat-value">{final_n_cells}</span>
                        <span class="stat-label">Total Cells</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-value">{final_conf:.1%}</span>
                        <span class="stat-label">Model Confidence</span>
                    </div>
                </div>
            </div>

            <div class="top-grid">
                <div class="card"><h4>Temporal Context</h4>{div_temp}</div>
                <div class="card"><h4>Cross-Slice Local Minima</h4>{global_summary_table}</div>
            </div>
            
            <div class="tab">{tab_buttons}</div>
            {slice_sections}

            <script id="plotly-data" type="application/json">
                {js_data_dump}
            </script>

            <script>
                var plotDataStore = {{}};
                try {{
                    var rawData = document.getElementById('plotly-data').textContent;
                    plotDataStore = JSON.parse(rawData);
                }} catch(e) {{
                    console.error("Failed to parse Plotly JSON data:", e);
                }}

                function renderPlots(sId, rank) {{
                    if (typeof Plotly === 'undefined') return;
                    var data = plotDataStore[sId][rank];
                    var config = {{responsive: true}};
                    Plotly.react('align-container-' + sId, data.align.data, data.align.layout, config);
                    Plotly.react('conf-container-' + sId, data.conf.data, data.conf.layout, config);
                }}

                function openSlice(evt, sId) {{
                    var i, content, links;
                    content = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < content.length; i++) {{ content[i].style.display = "none"; }}
                    
                    links = document.getElementsByClassName("tablinks");
                    for (i = 0; i < links.length; i++) {{ links[i].className = links[i].className.replace(" active", ""); }}
                    
                    var currentTab = document.getElementById("slice-" + sId);
                    currentTab.style.display = "block";
                    evt.currentTarget.className += " active";
                    
                    setTimeout(function() {{
                        var currentRank = document.getElementById('select-' + sId).value;
                        renderPlots(sId, currentRank);

                        var plotlyPlots = currentTab.querySelectorAll('.js-plotly-plot');
                        plotlyPlots.forEach(function(plot) {{
                            if (window.Plotly) {{ Plotly.Plots.resize(plot); }}
                        }});
                    }}, 50);
                }}

                document.addEventListener("DOMContentLoaded", function() {{
                    var winningSlice = "{winning_slice_id}";
                    var winningRank = "{winning_rank}";
                    
                    setTimeout(function() {{
                        renderPlots(winningSlice, winningRank);
                    }}, 150);
                }});
            </script>
        </body>
        </html>
        """
        with open(output_path, "w", encoding="utf-8") as f: 
            f.write(html_template)
        return output_path