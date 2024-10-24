function separation_across_tasks_by_trial(tasks, tables, colors,save_to)
hs = [];
figure
legend_entry = [];
for s = 1:length(tasks)
    table = tables(s);
    table = table{1};
    tit = tasks(s);

    table.decision_timing = get_arr(table.decision_timing);
    table = table(table.decision_timing ~= 0, :);

    num_trials = height(table);

    legend_entry = [legend_entry; tit + " with n = " + string(num_trials) + " trials"];
    
    timing = table.decision_timing; 
    mean_timing = mean(timing);
    std_err_timing = std(timing) / length(timing);

    eng = get_arr(table.avg_eng);
    mean_eng = mean(eng);
    std_err_eng = std(eng) / length(eng);

    h = scatter(mean_timing, mean_eng, 20, colors(s), 'filled');
    hold on 
    plot([mean_timing, mean_timing]', [-std_err_eng, std_err_eng]'+mean_eng', colors(s))
    hold on 
    plot([-std_err_timing, std_err_timing]'+mean_timing', [mean_eng, mean_eng]', colors(s))
    hold on

    %{
    h = scatter3(mean_max, mean_shift, mean_slope, 1, C(s), 'filled');
    hold on
    plot3([mean_max,mean_max]', [mean_shift,mean_shift]', [-slope_std_error,slope_std_error]'+mean_slope', C(s))  
    hold on
    plot3([mean_max,mean_max]', [-shift_std_error,shift_std_error]'+mean_shift', [mean_slope,mean_slope]', C(s))  
    hold on
    plot3([-max_std_error,max_std_error]'+mean_max, [mean_shift,mean_shift]', [mean_slope,mean_slope]', C(s))  
    hold on
    legend(story_type)
    hold on
    %}
    hs = [hs; h];
end

title("Timing vs Avg Eng By Trial")
legend(hs, legend_entry);
xlabel('Timing')
ylabel('Avg Eng')
fighandle = gcf;
set(gcf,'renderer','Painters')
saveas(fighandle,strcat(save_to,'2d_plot_by_trials.fig'))


end