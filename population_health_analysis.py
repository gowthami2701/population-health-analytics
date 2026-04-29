"""
Population Health Analytics - Chronic Disease Surveillance
CDC PLACES Data Analysis | Healthcare Data Analyst Portfolio
Author: Gowthami Vasamsetti

Analyzes chronic disease burden across US counties using CDC PLACES data.
Identifies high-risk populations and geographic health disparities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── SYNTHETIC CDC PLACES DATA GENERATOR ──────────────────────────────────────

STATES = {
    'CA': 'California', 'TX': 'Texas', 'FL': 'Florida', 'NY': 'New York',
    'PA': 'Pennsylvania', 'IL': 'Illinois', 'OH': 'Ohio', 'GA': 'Georgia',
    'NC': 'North Carolina', 'MI': 'Michigan', 'MD': 'Maryland', 'VA': 'Virginia',
}

def generate_county_health_data(n_counties: int = 500) -> pd.DataFrame:
    """Generate synthetic CDC PLACES-style county health data."""
    
    print("Generating county-level health data (CDC PLACES format)...")
    
    state_list = list(STATES.keys())
    
    # Base health profiles (some counties are healthier than others)
    county_health_index = np.random.beta(5, 3, n_counties)  # 0-1, higher = healthier
    
    # Socioeconomic deprivation index (correlated with health outcomes)
    deprivation = 1 - county_health_index + np.random.normal(0, 0.1, n_counties)
    deprivation = deprivation.clip(0, 1)
    
    records = []
    for i in range(n_counties):
        state = np.random.choice(state_list)
        hi = county_health_index[i]
        dep = deprivation[i]
        
        # Population (log-normal distribution like real counties)
        population = int(np.random.lognormal(10.5, 1.2, 1)[0].clip(5000, 10000000))
        
        # Chronic disease prevalence rates (% of adults)
        # Higher deprivation → higher disease rates
        diabetes_prev   = (0.08 + dep * 0.12 + np.random.normal(0, 0.02)).clip(0.04, 0.25)
        hypertension    = (0.28 + dep * 0.18 + np.random.normal(0, 0.03)).clip(0.18, 0.60)
        obesity         = (0.30 + dep * 0.20 + np.random.normal(0, 0.03)).clip(0.15, 0.65)
        copd            = (0.05 + dep * 0.08 + np.random.normal(0, 0.01)).clip(0.02, 0.20)
        chd             = (0.06 + dep * 0.07 + np.random.normal(0, 0.01)).clip(0.02, 0.18)
        stroke          = (0.03 + dep * 0.04 + np.random.normal(0, 0.008)).clip(0.01, 0.12)
        asthma          = (0.09 + dep * 0.06 + np.random.normal(0, 0.015)).clip(0.04, 0.22)
        depression      = (0.18 + dep * 0.10 + np.random.normal(0, 0.02)).clip(0.08, 0.40)
        ckd             = (0.03 + dep * 0.05 + np.random.normal(0, 0.01)).clip(0.01, 0.14)
        cancer          = (0.06 + (1-hi) * 0.04 + np.random.normal(0, 0.01)).clip(0.02, 0.16)
        
        # Social determinants of health (SDOH)
        uninsured_rate  = (0.08 + dep * 0.20 + np.random.normal(0, 0.02)).clip(0.01, 0.35)
        poverty_rate    = (0.10 + dep * 0.25 + np.random.normal(0, 0.03)).clip(0.02, 0.50)
        no_hs_diploma   = (0.08 + dep * 0.20 + np.random.normal(0, 0.02)).clip(0.01, 0.45)
        food_insecurity = (0.10 + dep * 0.20 + np.random.normal(0, 0.02)).clip(0.03, 0.45)
        
        # Healthcare access
        preventive_care_use  = (0.75 - dep * 0.30 + np.random.normal(0, 0.04)).clip(0.30, 0.95)
        physician_ratio      = (250 - dep * 150 + np.random.normal(0, 25)).clip(50, 500)  # per 100k
        
        # Health behaviors
        smoking_rate    = (0.14 + dep * 0.14 + np.random.normal(0, 0.02)).clip(0.05, 0.38)
        physical_inact  = (0.22 + dep * 0.18 + np.random.normal(0, 0.02)).clip(0.10, 0.55)
        
        # Composite scores
        chronic_burden  = np.mean([diabetes_prev, hypertension, obesity, copd, chd]) * 100
        sdoh_risk_score = np.mean([uninsured_rate, poverty_rate, food_insecurity]) * 100
        
        records.append({
            'county_fips':              f"{np.random.randint(10000,99999):05d}",
            'state_abbr':               state,
            'state_name':               STATES[state],
            'county_name':              f"County {i+1}",
            'population':               population,
            'county_health_index':      round(hi, 3),
            'deprivation_index':        round(dep, 3),
            # Chronic disease prevalence (% adults)
            'diabetes_pct':             round(diabetes_prev * 100, 2),
            'hypertension_pct':         round(hypertension * 100, 2),
            'obesity_pct':              round(obesity * 100, 2),
            'copd_pct':                 round(copd * 100, 2),
            'coronary_heart_disease_pct': round(chd * 100, 2),
            'stroke_pct':               round(stroke * 100, 2),
            'asthma_pct':               round(asthma * 100, 2),
            'depression_pct':           round(depression * 100, 2),
            'chronic_kidney_pct':       round(ckd * 100, 2),
            'cancer_pct':               round(cancer * 100, 2),
            # SDOH
            'uninsured_pct':            round(uninsured_rate * 100, 2),
            'poverty_pct':              round(poverty_rate * 100, 2),
            'no_hs_diploma_pct':        round(no_hs_diploma * 100, 2),
            'food_insecurity_pct':      round(food_insecurity * 100, 2),
            # Healthcare access
            'preventive_care_pct':      round(preventive_care_use * 100, 2),
            'physicians_per_100k':      round(physician_ratio, 1),
            # Behaviors
            'smoking_pct':              round(smoking_rate * 100, 2),
            'physical_inactivity_pct':  round(physical_inact * 100, 2),
            # Composite
            'chronic_burden_score':     round(chronic_burden, 2),
            'sdoh_risk_score':          round(sdoh_risk_score, 2),
        })
    
    df = pd.DataFrame(records)
    print(f"✅ Generated {len(df)} county records across {df['state_abbr'].nunique()} states")
    return df


# ── ANALYSIS FUNCTIONS ────────────────────────────────────────────────────────

def health_equity_analysis(df: pd.DataFrame):
    """Identify health disparities and high-need counties."""
    
    print("\n" + "="*65)
    print("POPULATION HEALTH ANALYTICS REPORT")
    print("="*65)
    
    # National averages
    print("\n📊 NATIONAL CHRONIC DISEASE AVERAGES")
    print("-"*45)
    metrics = ['diabetes_pct', 'hypertension_pct', 'obesity_pct', 
               'copd_pct', 'depression_pct', 'uninsured_pct']
    for m in metrics:
        avg = df[m].mean()
        worst = df[m].max()
        best = df[m].min()
        print(f"  {m.replace('_pct','').replace('_',' ').title():<25} "
              f"Avg: {avg:5.1f}%  Range: {best:.1f}% - {worst:.1f}%")
    
    # County clustering (identify health archetypes)
    print("\n🗂️ COUNTY HEALTH ARCHETYPES (K-Means Clustering)")
    print("-"*55)
    
    cluster_features = [
        'diabetes_pct', 'hypertension_pct', 'obesity_pct', 
        'uninsured_pct', 'poverty_pct', 'preventive_care_pct',
        'smoking_pct', 'physical_inactivity_pct'
    ]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['health_cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_labels = {}
    cluster_summary = df.groupby('health_cluster')[cluster_features + ['population']].mean()
    
    for cluster_id in range(4):
        row = cluster_summary.loc[cluster_id]
        if row['diabetes_pct'] > 14 and row['uninsured_pct'] > 15:
            label = 'High-Risk / Underserved'
        elif row['preventive_care_pct'] > 78 and row['obesity_pct'] < 32:
            label = 'Healthy / Well-Served'
        elif row['obesity_pct'] > 38:
            label = 'Metabolic Risk'
        else:
            label = 'Moderate Risk / Average Access'
        
        cluster_labels[cluster_id] = label
        n = (df['health_cluster'] == cluster_id).sum()
        print(f"\n  Cluster {cluster_id}: {label} ({n} counties)")
        print(f"    Diabetes: {row['diabetes_pct']:.1f}%  "
              f"Uninsured: {row['uninsured_pct']:.1f}%  "
              f"Preventive Care: {row['preventive_care_pct']:.1f}%")
    
    df['cluster_label'] = df['health_cluster'].map(cluster_labels)
    
    # Top 10 highest burden counties
    print("\n🚨 TOP 10 HIGHEST CHRONIC DISEASE BURDEN COUNTIES")
    print("-"*65)
    top10 = df.nlargest(10, 'chronic_burden_score')[
        ['county_name', 'state_abbr', 'chronic_burden_score', 
         'sdoh_risk_score', 'uninsured_pct', 'population']
    ]
    for _, row in top10.iterrows():
        print(f"  {row['county_name']}, {row['state_abbr']}  |  "
              f"Chronic Burden: {row['chronic_burden_score']:.1f}  |  "
              f"Uninsured: {row['uninsured_pct']:.1f}%  |  "
              f"Pop: {int(row['population']):,}")
    
    # SDOH-Disease correlation analysis
    print("\n📈 SOCIAL DETERMINANTS → DISEASE CORRELATIONS")
    print("-"*55)
    sdoh_vars = ['poverty_pct', 'uninsured_pct', 'food_insecurity_pct']
    disease_vars = ['diabetes_pct', 'hypertension_pct', 'obesity_pct']
    
    for disease in disease_vars:
        corrs = []
        for sdoh in sdoh_vars:
            corr = df[sdoh].corr(df[disease])
            corrs.append(f"{sdoh.replace('_pct','')}: r={corr:.3f}")
        print(f"  {disease.replace('_pct','').title():<20} | {' | '.join(corrs)}")
    
    # Intervention priority matrix
    print("\n🎯 INTERVENTION PRIORITY MATRIX")
    print("-"*55)
    
    # Counties with HIGH disease burden + HIGH SDOH risk = TOP PRIORITY
    high_burden = df['chronic_burden_score'] > df['chronic_burden_score'].quantile(0.75)
    high_sdoh = df['sdoh_risk_score'] > df['sdoh_risk_score'].quantile(0.75)
    low_access = df['physicians_per_100k'] < df['physicians_per_100k'].quantile(0.25)
    
    priority_1 = df[high_burden & high_sdoh & low_access]
    priority_2 = df[high_burden & high_sdoh & ~low_access]
    priority_3 = df[high_burden & ~high_sdoh]
    
    total_pop_p1 = priority_1['population'].sum()
    
    print(f"  🔴 Priority 1 (High Disease + High SDOH + Low Access): "
          f"{len(priority_1)} counties | {total_pop_p1/1e6:.2f}M people")
    print(f"  🟠 Priority 2 (High Disease + High SDOH):              "
          f"{len(priority_2)} counties | {priority_2['population'].sum()/1e6:.2f}M people")
    print(f"  🟡 Priority 3 (High Disease, Better Access):           "
          f"{len(priority_3)} counties | {priority_3['population'].sum()/1e6:.2f}M people")
    
    # State-level summary
    print("\n🗺️  STATE-LEVEL HEALTH RANKINGS")
    print("-"*55)
    state_summary = df.groupby('state_name').agg(
        Counties=('county_fips', 'count'),
        Avg_Diabetes=('diabetes_pct', 'mean'),
        Avg_Obesity=('obesity_pct', 'mean'),
        Avg_Uninsured=('uninsured_pct', 'mean'),
        Avg_Chronic_Burden=('chronic_burden_score', 'mean')
    ).round(2).sort_values('Avg_Chronic_Burden', ascending=False)
    
    print(state_summary.to_string())
    
    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Generate and analyze data
    df = generate_county_health_data(500)
    df = health_equity_analysis(df)
    
    # Save full dataset for Tableau
    df.to_csv('data/county_health_data.csv', index=False)
    
    # Save priority counties for targeted interventions
    priority = df[
        (df['chronic_burden_score'] > df['chronic_burden_score'].quantile(0.75)) &
        (df['sdoh_risk_score'] > df['sdoh_risk_score'].quantile(0.75))
    ].sort_values('chronic_burden_score', ascending=False)
    priority.to_csv('outputs/priority_intervention_counties.csv', index=False)
    
    print(f"\n💾 Saved county_health_data.csv ({len(df)} records)")
    print(f"💾 Saved priority_intervention_counties.csv ({len(priority)} counties)")
    print("\n🏁 Load county_health_data.csv into Tableau for geographic visualization!")
    print("   Recommended viz: Filled map by chronic_burden_score + scatter plot SDOH vs disease")
