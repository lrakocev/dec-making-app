function bar_plots_within_task_by_subject_hr(table, tit, cost_or_rew,save_to)
figure
table = table(str2double(table.number_of_timesteps(:)) > 9, :);
table.real_r = str2num(cell2mat(table.real_r));
table.real_c = str2num(cell2mat(table.real_c));

subjects = unique(table.subjectidnumber);
num_subjects = length(subjects);

y_bars = [];
y_errs = [];
for l = 1:4
    if cost_or_rew == "reward"
        curr_table = table(table.real_c == l, :);
    else
        curr_table = table(table.real_r == l, :);
    end
        
    curr_level = [];
    for s = 1:length(subjects)
        subid = subjects(s);
        sub_table = curr_table(curr_table.subjectidnumber == string(subid{1}), :);
        
        if ~isempty(sub_table)
            y = get_arr(sub_table.percent_max_hr);
            curr_level = [curr_level; mean(y)];
        end
    end

    y_bars = [y_bars; mean(curr_level, 'omitnan')];
    y_errs = [y_errs; std(curr_level, 'omitnan')/sum(~isnan(curr_level))];

end

if cost_or_rew == "reward"
    xlabelstr = "cost";
else
    xlabelstr = "reward";
end
ylabelstr = "% max above eng";

xs = [1,2,3,4];
errorbar(xs, y_bars, y_errs, y_errs)

xlabel(xlabelstr)
ylabel(ylabelstr)
title(tit + " with n = " + string(num_subjects) + " subjects")
savefig(strcat(save_to,tit{1},"_by_subject_",strrep(xlabelstr," ", "_"),"_", strrep(ylabelstr," ", "_")))

end