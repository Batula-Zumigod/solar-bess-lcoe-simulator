import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime
from io import BytesIO
import zipfile

# Import simulation engine
from simulation_engine import (
    WeatherDataDownloader, PVSystem, BatterySystem, 
    HybridEnergySimulator, FinancialAnalyzer, generate_comprehensive_excel
)
from html_report_generator import generate_html_report
import urllib.request

def get_plotly_js():
    """Download plotly.js once and cache it in session state"""
    if 'plotly_js' not in st.session_state:
        url = "https://cdn.plot.ly/plotly-2.27.0.min.js"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                st.session_state.plotly_js = r.read().decode('utf-8')
        except Exception:
            st.session_state.plotly_js = None  # fallback to CDN
    return st.session_state.plotly_js

st.set_page_config(page_title="Hybrid Energy LCOE Optimizer", layout="wide", page_icon="⚡")

st.title("⚡ Hybrid PV + BESS LCOE Optimization Tool")
st.markdown("**Prefeasibility analysis for finding the cheapest LCOE of renewable energy combinations**")

# ============================================================================
# SIDEBAR: PARAMETERS FILE UPLOAD
# ============================================================================
with st.sidebar:
    st.header("📁 Parameters Configuration")
    
    uploaded_params = st.file_uploader("Upload Parameters CSV", type=['csv'], 
                                      help="Upload your Parameters_Template.csv file")
    
    default_params_path = "Parameters_Template.csv"
    params = {}
    
    if uploaded_params:
        try:
            params_df = pd.read_csv(uploaded_params)
            st.success(f"✓ Loaded {len(params_df)} parameters from uploaded file")
            
            for _, row in params_df.iterrows():
                category = row['Category']
                param_name = row['Parameter']
                value = row['Value']
                
                if category not in params:
                    params[category] = {}
                params[category][param_name] = value
                
        except Exception as e:
            st.error(f"Error loading parameters file: {e}")
            params = {}
    else:
        if os.path.exists(default_params_path):
            try:
                params_df = pd.read_csv(default_params_path)
                st.info("Using default parameters file")
                
                for _, row in params_df.iterrows():
                    category = row['Category']
                    param_name = row['Parameter']
                    value = row['Value']
                    
                    if category not in params:
                        params[category] = {}
                    params[category][param_name] = value
                    
            except Exception as e:
                st.error(f"Error loading default parameters: {e}")
                params = {}

# Parameter getter function
PARAM_TYPES = {
    'Latitude': float, 'Longitude': float,'Time_Step': float,
    'Panel_Tilt': float, 'Panel_Azimuth': float, 'PV_Min_kWp': float,
    'PV_Max_kWp': float, 'PV_Step_kWp': float, 'Temperature_Coefficient': float,
    'PV_System_Losses': float, 'Inverter_Efficiency': float, 'ACDC_Ratio': float,
    'PV_Degradation': float, 'PV_LID': float,
    'BESS_Min_kWh': float, 'BESS_Max_kWh': float,
    'BESS_Hours': float, 'BESS_Step_kWh': float, 'Round_Trip_Efficiency': float, 'SOC_Min': float,
    'SOC_Max': float,
    'Replacement_80': bool, 'Replacement_70': bool, 'Replacement_60': bool, 'Never_Replace': bool,
    'Backup_Type': str, 'Diesel_Fuel_Consumption': float,
    'Diesel_Fuel_Price': float, 'Noon_Grid_Tariff': float, 'Peak_Grid_Tariff': float, 'Night_Grid_Tariff': float,
    'Grid_Limit_Min_kW': float,
    'Grid_Limit_Max_kW': float,
    'Grid_Limit_Step_kW': float,
    'Simulation_Years': int, 'Discount_Rate': float,
    'CAPEX_PV_Module': float, 'CAPEX_Mounting': float,
    'CAPEX_DC_BOS': float, 'CAPEX_SCADA_Base': float,
    'CAPEX_SCADA_per_kWp': float, 'CAPEX_BESS_Battery': float,
    'CAPEX_BESS_PCS': float, 'CAPEX_BESS_BOS': float, 'CAPEX_Diesel': float,
    'CAPEX_Grid': float, 'CAPEX_Other': float, 'OPEX_PV': float,
    'OPEX_BESS': float, 'OPEX_Diesel': float, 'OPEX_Grid': float,
    'Diesel_Escalation': float, 'Grid_Escalation': float,
    'LCOE_Threshold': float, 'Min_Self_Sufficiency': float,
}

def get_param(category, param_name):
    """Get parameter value from loaded parameters CSV"""
    if not params:
        st.error(f"❌ MISSING PARAMETER: No parameters file loaded. "
                 f"Cannot find [{category}] → {param_name}. "
                 f"Upload a Parameters CSV file first.")
        st.stop()

    if category not in params:
        st.error(f"❌ MISSING PARAMETER: Category '{category}' not found in parameters file. "
                 f"Cannot find [{category}] → {param_name}. "
                 f"Check your Parameters CSV.")
        st.stop()

    if param_name not in params[category]:
        st.error(f"❌ MISSING PARAMETER: '{param_name}' not found under [{category}] in parameters file. "
                 f"Check your Parameters CSV.")
        st.stop()

    raw_value = params[category][param_name]
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        st.error(f"❌ EMPTY PARAMETER: [{category}] → {param_name} exists but has no value. "
                 f"Fill it in your Parameters CSV.")
        st.stop()

    value_str = str(raw_value).strip()

    if param_name in PARAM_TYPES:
        data_type = PARAM_TYPES[param_name]
    else:
        if value_str.upper() in ['TRUE', 'FALSE']:
            data_type = bool
        elif '.' in value_str and value_str.replace('.', '').replace('-', '').isdigit():
            data_type = float
        elif value_str.replace('-', '').isdigit():
            data_type = int
        else:
            data_type = str

    try:
        if data_type == bool:
            return value_str.upper() == 'TRUE'
        elif data_type == float:
            return float(value_str)
        elif data_type == int:
            return int(float(value_str))
        elif data_type == str:
            return value_str
        else:
            return data_type(value_str)
    except (ValueError, TypeError) as e:
        st.error(f"❌ INVALID PARAMETER VALUE: [{category}] → {param_name} = '{value_str}' "
                 f"cannot be converted to {data_type.__name__}. Error: {e}")
        st.stop()

# ============================================================================
# SIDEBAR: ENERGY SOURCE SELECTION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Energy System Configuration")
    TimeStep = st.number_input("Time Step (Minutes). For the time being only use 60!", 
                              value=get_param('Location', 'Time_Step'),
                              format="%.2f")/60
    
    st.subheader("🔌 Energy Sources")
    col1, col2, col3 = st.columns(3)
    with col1:
        pv_enabled = st.checkbox("🌞 Solar PV", 
                                value=get_param('PV_System', 'PV_Min_kWp') > 0,
                                help="Include photovoltaic generation",
                                key="pv_enabled")
    with col3:
        bess_enabled = st.checkbox("🔋 BESS", 
                                  value=get_param('BESS', 'BESS_Min_kWh') > 0,
                                  help="Include battery energy storage",
                                   key="bess_enabled")
    
    if not pv_enabled:
        st.error("⚠️ Please select at least one generation source (PV)!")
        st.stop()

# ============================================================================
# SIDEBAR: LOCATION & WEATHER DATA
# ============================================================================
with st.sidebar:
    st.header("📍 Location & Weather Data")
    
    latitude = st.number_input("Latitude (°N)", 
                              value=get_param('Location', 'Latitude'),
                              format="%.4f")
    longitude = st.number_input("Longitude (°E)", 
                               value=get_param('Location', 'Longitude'),
                               format="%.4f")
    
    auto_download_weather_PV = st.checkbox("Auto-download weather data", value=True,
                                       help="Automatically download TMY/ERA5 data for selected location",
                                        key="auto_download_weather_PV")

# ============================================================================
# SIDEBAR: LOAD PROFILE
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Load Profile")
    
    uploaded_load = st.file_uploader("Upload Load Data (Excel)", type=['xlsx', 'xls'],
                                     help="Excel file with 'Load_kW' column")
    
    load_df = None
    if uploaded_load:
        load_df = pd.read_excel(uploaded_load)
    
    if load_df is not None:
        if 'Load_kW' not in load_df.columns:
            st.error("Load file must contain 'Load_kW' column")
            load_df = None
        else:
            if len(load_df) > 8760:
                load_df = load_df.iloc[:8760]
                st.warning(f"Trimmed load data to 8760 hours")
            elif len(load_df) < 8760:
                st.error(f"Load data has only {len(load_df)} hours. Need 8760 hours.")
                load_df = None
            else:
                st.success(f"✓ Loaded {len(load_df)} hours of load data")
                avg_load = load_df['Load_kW'].mean()
                peak_load = load_df['Load_kW'].max()
                st.metric("Average Load", f"{avg_load:.1f} kW")
                st.metric("Peak Load", f"{peak_load:.1f} kW")

# ============================================================================
# SIDEBAR: PV SYSTEM PARAMETERS
# ============================================================================
if pv_enabled:
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚡ PV System Sizing")
        
        pv_tilt = st.number_input("Panel Tilt Angle (°)", 
                                 value=get_param('PV_System', 'Panel_Tilt'),
                                 min_value=0.0, max_value=90.0, step=1.0)
        pv_azimuth = st.number_input("Panel Azimuth (°)", 
                                    value=get_param('PV_System', 'Panel_Azimuth'),
                                    min_value=0.0, max_value=360.0, step=1.0)
        
        col1, col2 = st.columns(2)
        with col1:
            pv_min = int(get_param('PV_System', 'PV_Min_kWp'))
            pv_min = st.number_input("Min PV Size (kWp)", value=pv_min, step=50)
        with col2:
            pv_max = int(get_param('PV_System', 'PV_Max_kWp'))
            pv_max = st.number_input("Max PV Size (kWp)", value=pv_max, step=50)
            
        pv_step = int(get_param('PV_System', 'PV_Step_kWp')) 
        pv_step = st.number_input("PV Step Size (kWp)", value=pv_step, step=50)
        
        with st.expander("🔧 Advanced PV Parameters"):
            pv_degradation = st.number_input("PV Degradation (%/year)", 
                                           value=get_param('PV_System', 'PV_Degradation') * 100,
                                           step=0.1) / 100
            pv_lid = st.number_input("PV LID (%/year)", 
                                           value=get_param('PV_System', 'PV_LID') * 100,
                                           step=0.1) / 100
            temp_coeff = st.number_input("Temperature Coefficient (%/°C)", 
                                       value=get_param('PV_System', 'Temperature_Coefficient') * 100,
                                       step=0.05) / 100
            PV_System_Losses = st.number_input("PV System Losses (%)", 
                                          value=get_param('PV_System', 'PV_System_Losses') * 100,
                                          step=1.0) / 100
            inverter_efficiency = st.number_input("Inverter Efficiency (%)", 
                                                value=get_param('PV_System', 'Inverter_Efficiency') * 100,
                                                step=0.5) / 100
            acdc_ratio = st.number_input("AC-DC Ratio", 
                                                value=get_param('PV_System', 'ACDC_Ratio'),
                                                step=0.5)
else:
    # Set defaults if PV is disabled
    pv_min = pv_max = pv_step = 0
    pv_degradation = pv_lid = temp_coeff = PV_System_Losses = inverter_efficiency = acdc_ratio = 0

# ============================================================================
# SIDEBAR: BESS PARAMETERS
# ============================================================================
if bess_enabled:
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔋 BESS Sizing")
        
        col1, col2 = st.columns(2)
        with col1:
            bess_min = st.number_input("Min BESS (kWh)", 
                                      value=get_param('BESS', 'BESS_Min_kWh'))
        with col2:
            bess_max = st.number_input("Max BESS (kWh)", value=get_param('BESS', 'BESS_Max_kWh'))
            bess_step = st.number_input("BESS Step (kWh)", value=get_param('BESS', 'BESS_Step_kWh'))
        
        with st.expander("🔋 Advanced BESS Parameters"):
            bess_hours = st.number_input("BESS Duration (hours)", 
                                        value=get_param('BESS', 'BESS_Hours'),
                                        min_value=0.5, max_value=10.0, step=0.5,
                                        help="Battery discharge duration at max power")
            bess_efficiency = st.number_input("Round-trip Efficiency (%)", 
                                            value=get_param('BESS', 'Round_Trip_Efficiency') * 100,
                                            step=1.0) / 100
            soc_min = st.number_input("Min SoC (%)", 
                                     value=get_param('BESS', 'SOC_Min') * 100,
                                     step=5.0) / 100
            soc_max = st.number_input("Max SoC (%)", 
                                     value=get_param('BESS', 'SOC_Max') * 100,
                                     step=5.0) / 100
            
            replacement_options = []
            if get_param('BESS', 'Replacement_80'):
                replacement_options.append(80)
            if get_param('BESS', 'Replacement_70'):
                replacement_options.append(70)
            if get_param('BESS', 'Replacement_60'):
                replacement_options.append(60)
            
            replacement_thresholds = st.multiselect(
                "Replacement Thresholds (%)",
                [80, 70, 60, 50],
                default=replacement_options,
                help="Replace battery when capacity falls below these levels"
            )
            include_no_replacement = st.checkbox("Include 'Never Replace' option", 
                                               value=get_param('BESS', 'Never_Replace'))
else:
    bess_min = bess_max = bess_step = 0
    bess_hours = bess_efficiency = soc_min = soc_max = 0
    replacement_thresholds = []
    include_no_replacement = False

# ============================================================================
# SIDEBAR: BACKUP SYSTEM WITH UNIVERSAL GRID LIMIT
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("🔌 Backup System")
    
    backup_type_from_param = get_param('Backup', 'Backup_Type')
    backup_options = ["Grid", "Diesel", "Grid+Diesel", "None"]
    backup_default_idx = backup_options.index(backup_type_from_param) if backup_type_from_param in backup_options else 1
    
    backup_type = st.radio("Select Backup Type", 
                          backup_options,
                          index=backup_default_idx,
                          help="Grid: Grid with limit\nDiesel: Diesel generator\nGrid+Diesel: Grid + diesel backup")
    
    if backup_type in ["Grid", "Grid+Diesel"]:
        st.info("⚡ **Grid Capacity Limit Sweep**")
        col1, col2 = st.columns(2)
        with col1:
            grid_limit_min = st.number_input(
                "Min Grid Limit (kW)", 
                value=float(get_param('Backup', 'Grid_Limit_Min_kW')),
                min_value=0.0,
                help="Set to 0 for NO grid"
            )
        with col2:
            grid_limit_max = st.number_input(
                "Max Grid Limit (kW)", 
                value=float(get_param('Backup', 'Grid_Limit_Max_kW')),
                min_value=0.0,
            )
        
        grid_limit_step = st.number_input(
            "Grid Limit Step (kW)", 
            value=float(get_param('Backup', 'Grid_Limit_Step_kW')),
            min_value=10.0,
        )
    
    # Grid tariffs (for Grid or Grid+Diesel)
    if backup_type in ["Grid", "Grid+Diesel"]:
        noon_grid_tariff = st.number_input("06:00-17:00 Energy Tariff ($/kWh)", 
                                     value=get_param('Backup', 'Noon_Grid_Tariff'))
        peak_grid_tariff = st.number_input("17:00-22:00 Energy Tariff ($/kWh)", 
                                     value=get_param('Backup', 'Peak_Grid_Tariff'))
        night_grid_tariff = st.number_input("22:00-06:00 Energy Tariff ($/kWh)", 
                                     value=get_param('Backup', 'Night_Grid_Tariff'))
    else:
        noon_grid_tariff = 0
        peak_grid_tariff = 0
        night_grid_tariff = 0
    
    # Diesel parameters (for Diesel or Grid+Diesel)
    if backup_type in ["Diesel", "Grid+Diesel"]:
        diesel_fuel_consumption = st.number_input("Fuel Consumption (L/kWh)", 
                                                 value=get_param('Backup', 'Diesel_Fuel_Consumption'))
        diesel_fuel_price = st.number_input("Fuel Price ($/L)", 
                                           value=get_param('Backup', 'Diesel_Fuel_Price'))
    else:
        diesel_fuel_consumption = 0
        diesel_fuel_price = 0

# ============================================================================
# SIDEBAR: SIMULATION PARAMETERS
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ Simulation Parameters")
    
    simulation_years = st.number_input("Simulation Years", 
                                      value=get_param('Financial', 'Simulation_Years'))
    discount_rate = st.number_input("Discount Rate (%)", 
                                   value=get_param('Financial', 'Discount_Rate') * 100,
                                   step=0.5) / 100
    


# ============================================================================
# SIDEBAR: FINANCIAL PARAMETERS
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("💰 Financial Parameters")
    
    with st.expander("CAPEX Parameters"):
        if pv_enabled:
            capex_pv_module = st.number_input("PV Modules ($/kW)", 
                                             value=get_param('Financial', 'CAPEX_PV_Module'))
            capex_mounting = st.number_input("Mounting ($/kW)", 
                                            value=get_param('Financial', 'CAPEX_Mounting'))
            capex_dc_bos = st.number_input("DC BOS ($/kW)", 
                                          value=get_param('Financial', 'CAPEX_DC_BOS'))
            capex_scada_base = st.number_input("SCADA Base ($)", 
                                              value=get_param('Financial', 'CAPEX_SCADA_Base'))
            capex_scada = st.number_input("SCADA ($/kW)", 
                                         value=get_param('Financial', 'CAPEX_SCADA_per_kWp'))
            capex_other = st.number_input("Other ($/kW)",
                                          value=get_param('Financial', 'CAPEX_Other'))
        else:
            capex_pv_module = 0
            capex_mounting = 0
            capex_dc_bos = 0
            capex_scada_base = 0
            capex_scada = 0
            capex_other = 0
        
        if bess_enabled:
            capex_bess_battery = st.number_input("Battery ($/kWh)", 
                                                value=get_param('Financial', 'CAPEX_BESS_Battery'))
            capex_bess_pcs = st.number_input("PCS ($/kWh)", 
                                            value=get_param('Financial', 'CAPEX_BESS_PCS'))
            capex_bess_bos = st.number_input("BESS BOS ($/kWh)", 
                                            value=get_param('Financial', 'CAPEX_BESS_BOS'))
        else:
            capex_bess_battery = 0
            capex_bess_pcs = 0
            capex_bess_bos = 0
        
        if backup_type in ["Diesel", "Grid+Diesel"]:
            capex_diesel = st.number_input("Diesel System ($/kW)", 
                                          value=get_param('Financial', 'CAPEX_Diesel'))
        else:
            capex_diesel = 0  
        
        if backup_type in ["Grid", "Grid+Diesel"]:
            capex_grid = st.number_input("Grid Connection ($)", 
                                        value=get_param('Financial', 'CAPEX_Grid'))
        else:
            capex_grid = 0  
    
    with st.expander("OPEX Parameters (Annual)"):
        if pv_enabled:
            opex_pv = st.number_input("PV O&M ($/kW)", 
                                     value=get_param('Financial', 'OPEX_PV'))
        else:
            opex_pv = 0
            
            
        if bess_enabled:
            opex_bess = st.number_input("BESS O&M ($/kWh)", 
                                       value=get_param('Financial', 'OPEX_BESS'))
        else:
            opex_bess = 0
            
        if backup_type in ["Diesel", "Grid+Diesel"]:
            opex_diesel = st.number_input("Diesel O&M ($/kW)", 
                                         value=get_param('Financial', 'OPEX_Diesel'))
        else:
            opex_diesel = 0
            
        if backup_type in ["Grid", "Grid+Diesel"]:
            opex_grid = st.number_input("Grid Contract Cost per kW ($)",
                                         value=get_param('Financial', 'OPEX_Grid'))
        else:
            opex_grid = 0

with st.sidebar:
    st.markdown("---")
    st.subheader("💰 Cost Escalation Rates")

    with st.expander("📈 Escalation Rates"):
        diesel_escalation = st.number_input("Diesel Price Escalation (%/year)",
                                         value=get_param('Financial', 'Diesel_Escalation'))/100

        grid_escalation = st.number_input("Grid Price Escalation (%/year)",
                                         value=get_param('Financial', 'Grid_Escalation'))/100

# ============================================================================
# FILTERS
# ============================================================================
with st.sidebar:
    st.markdown("---")
    st.subheader("⚖️ Filters & Export")
    
    lcoe_threshold = st.number_input("LCOE Threshold ($/kWh)", 
                                    value=get_param('Filters', 'LCOE_Threshold'))

    min_self_sufficiency_required = st.number_input("Self Sufficiency Required (%)", 
                                             value=get_param('Filters', 'Min_Self_Sufficiency'))

# ============================================================================
# MAIN: RUN SIMULATION BUTTON
# ============================================================================
st.markdown("---")
run_simulation = st.button("▶️ RUN SIMULATION", type="primary", use_container_width=True)

if run_simulation:
    # Clear any previous download data when starting a new simulation
    if 'current_zip_data' in st.session_state:
        st.session_state.current_zip_data = None
    
    # 🔧 FIX: Clear HTML report cache to regenerate with new simulation data
    if 'html_report_data' in st.session_state:
        st.session_state.html_report_data = None
    
    if load_df is None:
        st.error("Please upload a load profile file first!")
        st.stop()
    
    # Download weather data
    pv_weather_data = None
    if pv_enabled and auto_download_weather_PV:
        with st.spinner("🌤️ Downloading PV weather data..."):
            try:
                pv_weather_data = WeatherDataDownloader.download_tmy_pvgis(latitude, longitude)
                if len(pv_weather_data) != 8760:
                    pv_weather_data = pv_weather_data.iloc[:8760]
                st.success(f"✓ Downloaded {len(pv_weather_data)} hours of PV weather data")
            except Exception as e:
                st.error(f"❌ Failed to download PV weather data: {e}")
                st.stop()

        if 'datetime' in pv_weather_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(pv_weather_data['datetime']):
                pv_weather_data['datetime'] = pd.to_datetime(pv_weather_data['datetime'])
    
    # Prepare configuration ranges
    pv_sizes = list(range(int(pv_min), int(pv_max) + 1, int(pv_step))) if pv_enabled else [0]
    bess_sizes = list(range(int(bess_min), int(bess_max) + 1, int(bess_step))) if bess_enabled else [0]
    
    # FIX: Grid limit logic - if 0, means NO grid (set to empty list)
    if backup_type in ["Grid", "Grid+Diesel"]:
        if grid_limit_min == 0 and grid_limit_max == 0:
            grid_limit_sizes = []  # No grid configurations
        else:
            # Filter out 0 if it's in the range but not wanted
            grid_limit_sizes = [x for x in range(
                int(grid_limit_min), 
                int(grid_limit_max) + 1, 
                int(grid_limit_step)
            ) if x > 0]  # Only include positive values
    else:
        grid_limit_sizes = [0]  # For non-grid backup types, use 0 (will be ignored)
        
    replacement_thresholds_list = []
    if bess_enabled and replacement_thresholds:
        replacement_thresholds_list = [t/100 for t in replacement_thresholds]
        if include_no_replacement:
            replacement_thresholds_list.append(0.0)
    else:
        replacement_thresholds_list = [0.0]
    
    # Run simulations
    results_list = []
    total_sims = (len(pv_sizes) * len(bess_sizes) * 
              len(replacement_thresholds_list) * 
              max(len(grid_limit_sizes), 1))  # At least 1 iteration
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if 'failed_configs' not in st.session_state:
        st.session_state.failed_configs = []
    else:
        st.session_state.failed_configs = []

    sim_count = 0
    for pv_size in pv_sizes:
        for bess_size in bess_sizes:
            for replacement_threshold in replacement_thresholds_list:
                # Handle grid limit iterations
                grid_iterations = grid_limit_sizes if len(grid_limit_sizes) > 0 else [0]
                
                for grid_limit_kw in grid_iterations:
                    sim_count += 1
                    progress_bar.progress(sim_count / total_sims)
                    
                    # Display grid limit properly (0 = NO grid)
                    grid_display = "NO GRID" if grid_limit_kw == 0 else f"{grid_limit_kw:.0f}kW"
                    status_text.text(
                        f"Simulating: PV={pv_size}kW, BESS={bess_size}kWh, "
                        f"Grid={grid_display} [{sim_count}/{total_sims}]"
                    )
                    try:
                        # Initialize systems
                        pv_system = None
                        if pv_size > 0 and pv_weather_data is not None:
                            pv_system = PVSystem(
                                capacity_kwp=pv_size,
                                temp_coeff=temp_coeff,
                                PV_System_Losses=PV_System_Losses,
                                inverter_efficiency=inverter_efficiency,
                                acdc_ratio=acdc_ratio,
                                tilt=pv_tilt,
                                azimuth=pv_azimuth,
                                latitude=latitude,
                                longitude=longitude,
                                pv_lid=pv_lid,
                                pv_degradation=pv_degradation,
                                timezone='Asia/Ulaanbaatar',
                                TimeStep=TimeStep
                            )
                        
                        bess_system = None
                        if bess_size > 0:
                            bess_system = BatterySystem(
                                capacity_kwh=bess_size,
                                bess_hours=bess_hours,
                                bessefficiency=bess_efficiency,
                                soc_min=soc_min,
                                soc_max=soc_max,
                                TimeStep=TimeStep
                            )
                        
                        # Create simulator with REQUIRED grid_limit_kw
                        simulator = HybridEnergySimulator(
                            pv_system=pv_system,
                            bess_system=bess_system,
                            load_profile=load_df,
                            weather_data=pv_weather_data,
                            backup_type=backup_type,
                            diesel_fuel_consumption=diesel_fuel_consumption,
                            diesel_fuel_price=diesel_fuel_price,
                            noon_grid_tariff=noon_grid_tariff,
                            peak_grid_tariff=peak_grid_tariff,
                            night_grid_tariff=night_grid_tariff,
                            grid_limit_kw=grid_limit_kw,
                            TimeStep=TimeStep
                        )
                        
                        # Run simulation
                        results = simulator.simulate_multi_year(
                            num_years=simulation_years,
                            replacement_threshold=replacement_threshold,
                        )
                        
                        # Calculate financial metrics
                        capex_params_dict = {}
                        
                        if pv_enabled:
                            scada_cost = capex_scada_base + (capex_scada * pv_size)
                            capex_params_dict.update({
                                'pv_module': capex_pv_module * pv_size,
                                'mounting': capex_mounting * pv_size,
                                'dc_bos': capex_dc_bos * pv_size,
                                'scada': scada_cost,
                                'other': capex_other * pv_size
                            })
                        
                        if bess_enabled:
                            capex_params_dict.update({
                                'bess_battery': capex_bess_battery * bess_size,
                                'bess_pcs': capex_bess_pcs * bess_size,
                                'bess_bos': capex_bess_bos * bess_size
                            })
                        
                        if backup_type in ['Diesel', 'Grid+Diesel']:
                            capex_params_dict['diesel_system'] = capex_diesel * load_df['Load_kW'].max()
                        else:
                            capex_params_dict['diesel_system'] = 0

                        if backup_type in ['Grid', 'Grid+Diesel'] and grid_limit_kw > 0:
                            capex_params_dict['grid_system'] = capex_grid
                        else:
                            capex_params_dict['grid_system'] = 0
                        
                        opex_params_dict = {}
                        if pv_enabled:
                            opex_params_dict['pv_om'] = opex_pv * pv_size
                        if bess_enabled:
                            opex_params_dict['bess_om'] = opex_bess * bess_size
                        if backup_type in ['Diesel', 'Grid+Diesel']:
                            opex_params_dict['diesel_om'] = opex_diesel * load_df['Load_kW'].max()
                        else:
                            opex_params_dict['diesel_om'] = 0
                        if backup_type in ['Grid', 'Grid+Diesel'] and grid_limit_kw > 0:
                            opex_params_dict['grid_om'] = opex_grid * grid_limit_kw
                        else:
                            opex_params_dict['grid_om'] = 0
                        
                        financial_analyzer = FinancialAnalyzer(discount_rate=discount_rate)
                        lcoe_results = financial_analyzer.calculate_system_lcoe(
                            simulation_results=results,
                            capex_params=capex_params_dict,
                            opex_params=opex_params_dict,
                            project_years=simulation_years,
                            backup_type=backup_type,
                            diesel_escalation=diesel_escalation,
                            grid_escalation=grid_escalation
                        )
                        
                        # Apply LCOE threshold filter
                        if lcoe_results['lcoe_total'] > lcoe_threshold:
                            continue
                        
                        # Get self-sufficiency
                        self_sufficiency = results['avg_self_sufficiency_%']
                        renewable_fraction = results['avg_renewable_fraction'] * 100
                        
                        # Store results with NEW METRICS
                        first_year = results['annual_results'][0]
                        
                        # Calculate BESS cycle count
                        total_throughput_kwh = results['annual_results'][-1].get('battery_throughput_kwh', 0)
                        nominal_capacity = bess_size if bess_size > 0 else 1
                        bess_cycle_count = total_throughput_kwh / (2 * nominal_capacity)  # Full cycles
                        
                        # Calculate BESS efficiency losses
                        total_charged = sum(r.get('pv_to_battery_kwh', 0) + r.get('grid_to_battery_kwh', 0) + r.get('diesel_to_battery_kwh', 0) 
                                          for r in results['annual_results'])
                        total_discharged = sum(r.get('battery_to_load_kwh', 0) for r in results['annual_results'])
                        bess_efficiency_loss = total_charged - total_discharged if total_charged > 0 else 0
                        
                        # Total demand
                        total_demand_kwh = results['total_load_demand_kwh']
                        
                        results_list.append({
                            'PV_kWp': pv_size,
                            'BESS_kWh': bess_size,
                            'BESS_Power_kW': (bess_size / bess_hours) if bess_system is not None else 0,
                            'Grid_Limit_kW': grid_limit_kw,
                            'capex_params': capex_params_dict,
                            'opex_params': opex_params_dict,
                            'Round_Trip_Efficiency': bess_efficiency,
                            'Replacement_Threshold_%': replacement_threshold * 100 if replacement_threshold > 0 else 999,
                            'Self_Sufficiency_%': self_sufficiency,
                            'LCOE_Total_$/kWh': lcoe_results['lcoe_total'],
                            'LCOE_Renewable_$/kWh': lcoe_results['lcoe_renewable'],
                            'LCOE_PV_$/kWh': lcoe_results['lcoe_pv'],
                            'LCOE_BESS_$/kWh': lcoe_results['lcoe_bess'],
                            'LCOE_Backup_$/kWh': lcoe_results['lcoe_backup'],
                            'Total_CAPEX_$': lcoe_results['total_capex'],
                            'Renewable_Fraction_%': renewable_fraction,
                            'Battery_Replacements': results.get('battery_replacements', 0),
                            'PV_Energy_MWh': first_year['pv_generation_kwh'] / 1000,
                            'Backup_Energy_MWh': first_year['backup_kwh'] / 1000,
                            'Curtailed_Energy_MWh': first_year['curtailed_kwh'] / 1000,
                            'Curtailment_Rate_%': (
                                first_year['curtailed_kwh'] / first_year['pv_generation_kwh'] * 100
                                if first_year['pv_generation_kwh'] > 100 else 0.0
                            ),
                            'Generator_Spillage_MWh': first_year.get('generator_spillage_kwh', 0) / 1000,
                            'Unmet_Load_MWh': first_year.get('unmet_load_kwh', 0) / 1000,
                            'Total_Demand_MWh': total_demand_kwh / 1000/simulation_years,
                            'BESS_Cycle_Count': bess_cycle_count,
                            'BESS_Efficiency_Loss_MWh': bess_efficiency_loss / 1000,
                            'sim_results': results
                        })
                        
                    except Exception as e:
                        error_msg = f"PV={pv_size}kW, BESS={bess_size}kWh, Grid={grid_limit_kw}kW - Error: {str(e)}"
                        st.session_state.failed_configs.append(error_msg)
                        continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not results_list:
        st.error("No successful simulations. Please check your inputs.")
        
        # Show failed configurations for debugging
        if st.session_state.failed_configs:
            with st.expander("❌ Failed Configurations (click to see errors)"):
                for error in st.session_state.failed_configs[:10]:
                    st.text(error)
                if len(st.session_state.failed_configs) > 10:
                    st.text(f"... and {len(st.session_state.failed_configs) - 10} more errors")
        st.stop()
    
    results_df = pd.DataFrame(results_list)

    # ✅ RENAME FIRST (on full DataFrame)
    results_df = results_df.rename(columns={
        'LCOE_Total_/kWh': 'LCOE_Total_$/kWh',
        'LCOE_PV_/kWh': 'LCOE_PV_$/kWh',
        'LCOE_BESS_/kWh': 'LCOE_BESS_$/kWh',
        'LCOE_Backup_/kWh': 'LCOE_Backup_$/kWh',
    })

    # ✅ NOW FILTER (on renamed DataFrame)
    results_100pct = results_df[results_df['Self_Sufficiency_%'] >= min_self_sufficiency_required].copy()
    results_failed = results_df[results_df['Self_Sufficiency_%'] < min_self_sufficiency_required].copy()

    results_100pct = results_100pct.sort_values('LCOE_Total_$/kWh', ascending=True).reset_index(drop=True)
    results_failed = results_failed.sort_values('LCOE_Total_$/kWh', ascending=True).reset_index(drop=True)

    results_df_sorted = pd.concat([results_100pct, results_failed], ignore_index=True)
        
    # Find optimal configuration
    optimal_100pct = None
    if len(results_100pct) > 0:
        optimal_100pct = results_100pct.iloc[0]
    
    st.success(f"✅ Simulation Complete! {len(results_df)} configurations analyzed")
    
    # Generate grid limit summary
    grid_limit_summary = []
    unique_grid_limits = sorted(results_df['Grid_Limit_kW'].unique())
    
    for grid_limit in unique_grid_limits:
        grid_configs = results_df[results_df['Grid_Limit_kW'] == grid_limit]
        grid_configs_met = grid_configs[grid_configs['Self_Sufficiency_%'] >= min_self_sufficiency_required]
        
        if len(grid_configs_met) > 0:
            best_config = grid_configs_met.loc[grid_configs_met['LCOE_Total_$/kWh'].idxmin()]
            
            grid_limit_summary.append({
                'Grid_Limit_kW': grid_limit,
                'Status': '✅ FEASIBLE',
                'PV_kWp': best_config['PV_kWp'],
                'BESS_kWh': best_config['BESS_kWh'],
                'BESS_Power_kW': best_config['BESS_Power_kW'],
                'LCOE_$/kWh': best_config['LCOE_Total_$/kWh'],
                'Self_Sufficiency_%': best_config['Self_Sufficiency_%'],
                'Renewable_Fraction_%': best_config['Renewable_Fraction_%'],
                'Total_CAPEX_$': best_config['Total_CAPEX_$'],
                'Unmet_Load_MWh': best_config['Unmet_Load_MWh'],
                'Battery_Replacements': best_config['Battery_Replacements']
            })
        else:
            grid_limit_summary.append({
                'Grid_Limit_kW': grid_limit,
                'Status': '❌ IMPOSSIBLE',
                'PV_kWp': '-',
                'BESS_kWh': '-',
                'BESS_Power_kW': '-',
                'LCOE_$/kWh': '-',
                'Self_Sufficiency_%': '-',
                'Renewable_Fraction_%': '-',
                'Total_CAPEX_$': '-',
                'Unmet_Load_MWh': '-',
                'Battery_Replacements': '-'
            })
    
    grid_summary_df = pd.DataFrame(grid_limit_summary)
    
    # Initialize session state
    if 'current_zip_data' not in st.session_state:
        st.session_state.current_zip_data = None
    
    # ============================================================================
    # GENERATE HTML REPORT SEPARATELY (NOT IN ZIP)
    # ============================================================================
    if 'html_report_data' not in st.session_state:
        st.session_state.html_report_data = None

    if st.session_state.html_report_data is None and optimal_100pct is not None:
        with st.spinner("📊 Generating interactive HTML report..."):
            try:
                optimal_sim_results = optimal_100pct['sim_results']   # ← FIX: was NameError

                html_bytes = generate_html_report(
                    all_configs_df   = results_100pct.drop('sim_results', axis=1).copy(),
                    project_name     = f"Solar + BESS Analysis — {datetime.now().strftime('%Y-%m-%d')}",
                    discount_rate    = discount_rate,
                    optimal_capex    = optimal_100pct.get('capex_params'),
                    optimal_opex     = optimal_100pct.get('opex_params'),
                    load_hourly_data = load_df,
                    optimal_sim_results = optimal_sim_results,      # ← NEW
                    latitude         = latitude,                     # ← NEW
                    longitude        = longitude,                    # ← NEW
                    backup_type      = backup_type,                  # ← NEW
                    diesel_escalation = diesel_escalation,           # ← NEW
                    grid_escalation   = grid_escalation,             # ← NEW
                )

                st.session_state.html_report_data = html_bytes
                st.success("✅ HTML report generated!")

            except Exception as e:
                st.error(f"❌ Error generating HTML report: {e}")
                import traceback
                st.code(traceback.format_exc())


    # ============================================================================
    # AUTO-GENERATE ZIP (EXCEL FILES + HTML REPORT)
    # ============================================================================
    if st.session_state.current_zip_data is None:
        total_files = len(grid_summary_df) + 3  # Grid files + optimal + summary + HTML
        with st.spinner(f"🔄 Generating complete package ({total_files} files)... Please wait..."):
            try:
                zip_buffer = BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    
                    # 1. Generate Excel for each grid limit
                    file_counter = 1
                    for idx, row in grid_summary_df.iterrows():
                        grid_limit_kw = row['Grid_Limit_kW']
                        status = row['Status']
                        
                        if status == '✅ FEASIBLE':
                            grid_configs = results_df[results_df['Grid_Limit_kW'] == grid_limit_kw]
                            grid_configs_met = grid_configs[grid_configs['Self_Sufficiency_%'] >= min_self_sufficiency_required]
                            best_config = grid_configs_met.loc[grid_configs_met['LCOE_Total_$/kWh'].idxmin()]
                            
                            config_dict = {
                                'PV_kWp': best_config['PV_kWp'],
                                'BESS_kWh': best_config['BESS_kWh'],
                                'Replacement_Threshold_%': best_config['Replacement_Threshold_%'],
                                'Total_CAPEX_$': best_config['Total_CAPEX_$'],
                                'LCOE_PV_$/kWh': best_config['LCOE_PV_$/kWh'],
                                'LCOE_BESS_$/kWh': best_config['LCOE_BESS_$/kWh'],
                                'LCOE_Renewable_$/kWh': best_config['LCOE_Renewable_$/kWh'],
                                'LCOE_Total_$/kWh': best_config['LCOE_Total_$/kWh'],
                                'Renewable_Fraction_%': best_config['Renewable_Fraction_%'],
                                'Round_Trip_Efficiency': best_config['Round_Trip_Efficiency']
                            }
                            
                            excel_bytes = generate_comprehensive_excel(
                                optimal_config=config_dict,
                                simulation_results=best_config['sim_results'],
                                load_df=load_df,
                                simulation_years=int(simulation_years),
                                capex_params=best_config['capex_params'],
                                opex_params=best_config['opex_params'],
                                power_kw=best_config['BESS_Power_kW'],
                                discount_rate=discount_rate,
                                all_configs_df=None,
                                backup_type=backup_type,
                                latitude=latitude,
                                longitude=longitude,
                                diesel_escalation=diesel_escalation,
                                grid_escalation=grid_escalation,
                            )
                            
                            filename = f"{file_counter:02d}_Grid_{int(grid_limit_kw)}kW_LCOE_{best_config['LCOE_Total_$/kWh']:.4f}.xlsx"
                            zip_file.writestr(filename, excel_bytes)
                            file_counter += 1
                            
                        else:
                            impossible_df = pd.DataFrame([{
                                'Grid_Limit_kW': grid_limit_kw,
                                'Status': 'IMPOSSIBLE',
                                'Message': f'No configuration can achieve {min_self_sufficiency_required}% self-sufficiency with {grid_limit_kw} kW grid limit.'
                            }])
                            
                            impossible_buffer = BytesIO()
                            with pd.ExcelWriter(impossible_buffer, engine='openpyxl') as writer:
                                impossible_df.to_excel(writer, sheet_name='IMPOSSIBLE', index=False)
                            impossible_buffer.seek(0)
                            
                            filename = f"{file_counter:02d}_Grid_{int(grid_limit_kw)}kW_IMPOSSIBLE.xlsx"
                            zip_file.writestr(filename, impossible_buffer.getvalue())
                            file_counter += 1
                    
                    # 2. Overall optimal Excel
                    if optimal_100pct is not None:
                        optimal_config = {
                            'PV_kWp': optimal_100pct['PV_kWp'],
                            'BESS_kWh': optimal_100pct['BESS_kWh'],
                            'Replacement_Threshold_%': optimal_100pct['Replacement_Threshold_%'],
                            'Total_CAPEX_$': optimal_100pct['Total_CAPEX_$'],
                            'LCOE_PV_$/kWh': optimal_100pct['LCOE_PV_$/kWh'],
                            'LCOE_BESS_$/kWh': optimal_100pct['LCOE_BESS_$/kWh'],
                            'LCOE_Renewable_$/kWh': optimal_100pct['LCOE_Renewable_$/kWh'],
                            'LCOE_Total_$/kWh': optimal_100pct['LCOE_Total_$/kWh'],
                            'Renewable_Fraction_%': optimal_100pct['Renewable_Fraction_%'],
                            'Round_Trip_Efficiency': optimal_100pct['Round_Trip_Efficiency']
                        }

                        excel_overall_bytes = generate_comprehensive_excel(
                            optimal_config=optimal_config,
                            simulation_results=optimal_100pct['sim_results'],
                            load_df=load_df,
                            simulation_years=int(simulation_years),
                            capex_params=optimal_100pct['capex_params'],
                            opex_params=optimal_100pct['opex_params'],
                            power_kw=optimal_100pct['BESS_Power_kW'],
                            discount_rate=discount_rate,
                            all_configs_df=None,
                            backup_type=backup_type,
                            latitude=latitude,
                            longitude=longitude,
                            diesel_escalation=diesel_escalation,
                            grid_escalation=grid_escalation,
                        )
                        zip_file.writestr(f"{file_counter:02d}_OVERALL_OPTIMAL.xlsx", excel_overall_bytes)
                        file_counter += 1
                    
                    # 3. Summary Excel
                    summary_df = results_df_sorted.drop('sim_results', axis=1)
                    summary_buffer = BytesIO()
                    with pd.ExcelWriter(summary_buffer, engine='openpyxl') as writer:
                        summary_df.to_excel(writer, sheet_name='All_Configurations', index=False)
                    summary_buffer.seek(0)
                    zip_file.writestr(f"{file_counter:02d}_All_Configurations_Summary.xlsx", summary_buffer.getvalue())
                    file_counter += 1
                    
                    # 4. Add HTML Report to ZIP
                    if st.session_state.html_report_data is not None:
                        html_filename = f"{file_counter:02d}_Interactive_Report.html"
                        zip_file.writestr(html_filename, st.session_state.html_report_data)
                        st.success("✅ HTML report included in ZIP package!")

                
                zip_buffer.seek(0)
                st.session_state.current_zip_data = zip_buffer.getvalue()
                
                st.success("✅ Complete package generated successfully! (HTML + Excel files)")
                
            except Exception as e:
                st.error(f"❌ Error generating ZIP: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # SHOW ONLY DOWNLOAD BUTTON
    # ============================================================================
    # DOWNLOAD SECTION - COMBINED HTML + EXCEL ZIP
    # ============================================================================
    st.markdown("---")

    if st.session_state.current_zip_data is not None:
        
        st.info("""
        **📦 Complete Results Package Ready for Download**
        
        Your analysis package includes:
        - **Interactive HTML Report** with dropdown-controlled heatmaps and professional visualizations
        - **Excel Files** with detailed analysis for each grid configuration
        - **Summary Excel** with all configurations
        - **Optimal Configuration Excel** with complete financial breakdown
        
        Everything is packaged in a single ZIP file for your convenience!
        """)
        
        # Single Combined Download Button
        zip_filename = f"Complete_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_size_mb = len(st.session_state.current_zip_data) / (1024 * 1024)
        
        st.download_button(
            label=f"📥 Download Complete Package ({zip_size_mb:.1f} MB)",
            data=st.session_state.current_zip_data,
            file_name=zip_filename,
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )
        st.caption("✨ Includes interactive HTML report + all Excel files")
