import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTANT: Change this to the name of your generated CSV file
CSV_FILE = "final_results.csv"


def create_charts():
    df = pd.read_csv(CSV_FILE)
    sns.set_theme(style="whitegrid")

    # Because we ran 5 trials per scenario, we want to average them for the charts
    df_avg = df.groupby(
        ['Scenario', 'Algorithm', 'Total_Passengers'], as_index=False).mean()

    # 1. Bar Chart: Passengers Assigned
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_avg, x='Scenario', y='Passengers_Assigned',
                hue='Algorithm', palette=['#1f77b4', '#2ca02c'])
    demand = df_avg[['Scenario', 'Total_Passengers']].drop_duplicates()
    plt.scatter(x=range(len(demand)), y=demand['Total_Passengers'], color='red',
                marker='_', s=2000, linewidth=3, label='Total Demand', zorder=5)
    plt.title('Algorithm Performance: Total Passengers Assigned',
              fontsize=14, pad=15)
    plt.ylabel('Number of Passengers Assigned')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('chart_1_passengers.png', dpi=300)

    # 2. Line Chart: Runtime (Log Scale)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_avg, x='Total_Passengers', y='Runtime_MS', hue='Algorithm',
                 marker='o', markersize=10, linewidth=2.5, palette=['#1f77b4', '#2ca02c'])
    plt.yscale('log')
    plt.title('Computational Runtime vs. Passenger Demand (Log Scale)',
              fontsize=14, pad=15)
    plt.xlabel('Total Passenger Demand (Problem Size)')
    plt.ylabel('Runtime in Milliseconds (Log Scale)')
    plt.tight_layout()
    plt.savefig('chart_2_runtime.png', dpi=300)

    # 3. Scatter Plot: Occupancy Paradox
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_avg, x='Passengers_Assigned', y='Seat_Occupancy_Pct',
                    hue='Algorithm', style='Scenario', s=250, palette=['#1f77b4', '#2ca02c'])
    plt.title('The Paradox: Seat Occupancy vs. Passengers Assigned',
              fontsize=14, pad=15)
    plt.xlabel('Number of Passengers Assigned')
    plt.ylabel('Seat Occupancy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('chart_3_occupancy.png', dpi=300)

    print("[SUCCESS] Charts generated and saved as PNG files!")


if __name__ == "__main__":
    create_charts()
