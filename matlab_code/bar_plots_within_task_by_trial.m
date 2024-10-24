function bar_plots_within_task_by_trial(table, tit, cost_or_rew, save_to)
figure
table.decision_timing = get_arr(table.decision_timing);
table = table(table.decision_timing ~= 0, :);
n_trials = height(table);

table.real_r = str2num(cell2mat(table.real_r));
table.real_c = str2num(cell2mat(table.real_c));

y_bars = [];
y_errs = [];
xs = [];
for l = 1:4
    if cost_or_rew == "reward"
        curr_table = table(table.real_c == l, :);
    else
        curr_table = table(table.real_r == l, :);
    end
    if ~isempty(curr_table)
        xs = [xs; l];
        y = curr_table.decision_timing;
        y_bars = [y_bars; mean(y)];
        y_errs = [y_errs; std(y)/length(y)];
    end
end

if cost_or_rew == "reward"
    xlabelstr = "cost";
else
    xlabelstr = "reward";
end
ylabelstr = "timing";

%bar(xs, y_bars)
%hold on
errorbar(xs, y_bars, y_errs, y_errs)

xlabel(xlabelstr)
ylabel(ylabelstr)
title(tit + " with n = " + n_trials + " trials")
savefig(strcat(save_to,tit{1},"_by_trial_",strrep(xlabelstr," ", "_"),"_", strrep(ylabelstr," ", "_")))
end