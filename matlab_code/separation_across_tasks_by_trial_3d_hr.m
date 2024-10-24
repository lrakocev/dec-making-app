function separation_across_tasks_by_trial_3d_hr(tasks, tables, colors,save_to)
hs = [];
figure
legend_entry = [];
for s = 1:length(tasks)
    table = tables(s);
    table = table{1};
    table = table(str2double(table.number_of_timesteps(:)) > 9, :);

    tit = tasks(s);
    num_trials = height(table);

    legend_entry = [legend_entry; tit + " with n = " + string(num_trials) + " trials"];
    
    min_percent = get_arr(table.percent_min_hr); 
    mean_min_percent = mean(min_percent);
    std_err_min_percent = std(min_percent) / length(min_percent);

    direction = get_arr(table.direction);
    mean_direction = mean(direction);
    std_err_direction = std(direction) / length(direction);

    max_percent = get_arr(table.percent_max_hr);
    mean_max_percent = mean(max_percent);
    std_err_max_percent = std(max_percent) / length(max_percent);

    h = scatter3(mean_min_percent, mean_direction, mean_max_percent, 20, colors(s), 'filled');
    hold on 
    plot3([mean_min_percent, mean_min_percent]', [-std_err_direction, std_err_direction]'+mean_direction', [mean_max_percent, mean_max_percent]', colors(s))
    hold on 
    plot3([-std_err_min_percent, std_err_min_percent]'+mean_min_percent', [mean_direction, mean_direction]',[mean_max_percent, mean_max_percent]', colors(s))
    hold on
    plot3([mean_min_percent, mean_min_percent]', [mean_direction, mean_direction]',[-std_err_max_percent, std_err_max_percent]'+mean_max_percent', colors(s))
    hold on

    hs = [hs; h];
end

title("3D Separation of Heart Rate Data By Trial")
legend(hs, legend_entry);
xlabel('% min below avg')
ylabel('direction')
zlabel('% max above avg')
fighandle = gcf;
set(gcf,'renderer','Painters')
saveas(fighandle,strcat(save_to,'3d_plot_by_trials.fig'))


end