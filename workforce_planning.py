"""
Workforce Planning Optimization - Minimize Staffing

This script optimizes workforce scheduling to minimize the total number of staff
while meeting daily demand requirements, considering 5-day work weeks with 2 consecutive days off.
"""

import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import matplotlib.pyplot as plt
from itertools import chain, repeat


def ncycles(iterable, n):
    """Returns the sequence elements n times."""
    return chain.from_iterable(repeat(tuple(iterable), n))


def create_schedule_constraints():
    """Create circular lists for working days and days off."""
    n_days = list(range(7))
    n_days_c = list(ncycles(n_days, 3))

    # Working days (5 consecutive days)
    list_in = [[n_days_c[j] for j in range(i, i + 5)] for i in n_days_c]

    # Days off (2 consecutive days after 5 working days)
    list_excl = [[n_days_c[j] for j in range(i + 1, i + 3)] for i in n_days_c]

    return n_days, list_in, list_excl


def build_and_solve_model(n_days, list_excl, n_staff):
    """Build and solve the workforce optimization model."""
    model = LpProblem("Minimize_Staffing", LpMinimize)

    # Decision variables: number of workers starting their shift on each day
    x = LpVariable.dicts("shift", n_days, lowBound=0, cat="Integer")

    # Objective: minimize total number of staff
    model += lpSum([x[i] for i in n_days])

    # Constraints: meet daily staff demand
    for d, l_excl, staff in zip(n_days, list_excl, n_staff):
        model += lpSum([x[i] for i in n_days if i not in l_excl]) >= staff

    model.solve()

    return model, x


def extract_results(model, x, n_days, list_in, jours):
    """Extract and format optimization results."""
    # Extract shift assignments
    dct_work = {int(v.name[-1]): int(v.varValue) for v in model.variables()}

    # Build schedule matrix
    start_jours = ["Shift: " + j for j in jours]
    dict_sch = {}
    for day in dct_work.keys():
        dict_sch[day] = [dct_work[day] if i in list_in[day] else 0 for i in n_days]

    df_sch = pd.DataFrame(dict_sch).T
    df_sch.columns = jours
    df_sch.index = start_jours

    return df_sch, dct_work


def display_results(model, df_sch, n_staff, jours):
    """Display optimization results."""
    print("=" * 60)
    print("WORKFORCE PLANNING OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nStatus: {LpStatus[model.status]}")
    print(f"Total number of Staff = {int(value(model.objective))}")

    print("\n" + "-" * 60)
    print("SHIFT SCHEDULE (workers per shift per day)")
    print("-" * 60)
    print(df_sch.to_string())

    print("\n" + "-" * 60)
    print("DAILY STAFFING SUMMARY")
    print("-" * 60)

    df_staff = pd.DataFrame({"Days": jours, "Staff Demand": n_staff})
    df_staff = df_staff.set_index("Days")
    df_staff["Staff Supply"] = df_sch.sum(axis=0).values
    df_staff["Extra Resources"] = df_staff["Staff Supply"] - df_staff["Staff Demand"]

    print(df_staff.to_string())

    return df_staff


def plot_results(df_staff, save_path=None):
    """Plot workforce demand vs supply."""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x_pos = range(len(df_staff))
    width = 0.35

    ax1.bar(
        [p - width / 2 for p in x_pos],
        df_staff["Staff Demand"],
        width,
        label="Staff Demand",
        color="black",
    )
    ax1.bar(
        [p + width / 2 for p in x_pos],
        df_staff["Staff Supply"],
        width,
        label="Staff Supply",
        color="red",
    )

    ax1.set_xlabel("Day of the week")
    ax1.set_ylabel("Number of Workers")
    ax1.set_title("Workforce: Demand vs. Supply")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_staff.index)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        x_pos,
        df_staff["Extra Resources"],
        color="blue",
        linewidth=3,
        marker="o",
        label="Extra Resources",
    )
    ax2.set_ylabel("Extra Resources")
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"\nChart saved to: {save_path}")
    else:
        plt.show()

    return fig


def main():
    """Main function to run workforce optimization."""
    # Input data: staff needs per day (FTE)
    n_staff = [31, 45, 40, 40, 48, 30, 25]
    jours = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Create schedule constraints
    n_days, list_in, list_excl = create_schedule_constraints()

    # Build and solve model
    model, x = build_and_solve_model(n_days, list_excl, n_staff)

    # Extract results
    df_sch, dct_work = extract_results(model, x, n_days, list_in, jours)

    # Display results
    df_staff = display_results(model, df_sch, n_staff, jours)

    # Plot results
    plot_results(df_staff, save_path="workforce_chart.png")

    return df_sch, df_staff


if __name__ == "__main__":
    main()
