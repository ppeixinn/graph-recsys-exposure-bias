"""Result analysis: user-degree and item-popularity groups, recall breakdowns, figures."""
from .groups import item_groups, user_groups, group_info
from .breakdowns import topk, recall_by_item_group, recall_by_user_group, wins_and_failures
from .figures import plot_breakdowns
