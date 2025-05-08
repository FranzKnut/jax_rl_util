from dataclasses import dataclass
import re
from matplotlib import pyplot as plt
from matplotlib.colors import BASE_COLORS, TABLEAU_COLORS
import pandas as pd
import simple_parsing


@dataclass
class Arguments:
    """Arguments for the script."""

    file_name: str = "data/eval/Sweep3_halfcheetah.csv"
    column: str = "best_eval"
    by: list[str] = simple_parsing.list_field("policy_config.agent_type")
    title: str = "Boxplot"
    save: bool = True
    filters: str = ""  # see pandas query documentation


args = simple_parsing.parse(Arguments)
df = pd.read_csv(args.file_name, index_col=0)

# name runs
df = df.fillna("none")
print(df.head())
if args.filters:
    df = df.query(args.filters)
df["_name"] = pd.Series(
    map("_".join, df[args.by].values.astype(str).tolist()), index=df.index
)
# df.set_index("_name", inplace=True)
# df.plot.bar(
#     column=[args.column],
#     by=args.by,
# layout=(1, -1),
#     # rot=90,
# )
_group = df.groupby("_name", sort="_name")
y_err = _group[args.column].std()
print(_group[args.column].mean())
group_names = _group.groups.keys()
plt.bar(
    group_names,
    _group[args.column].mean(),
    color=TABLEAU_COLORS,
    yerr=y_err,
)
plt.ylim(0)
# plt.xticks([i for i in range(df["_name"].nunique())], list(df["_name"].unique()))
plt.title(args.title)
# plt.tight_layout()
if args.save:
    filename = re.sub(r'(?u)[^-\w.]', '_', args.title)
    plt.savefig(f"plots/{filename}.png")
plt.show()
