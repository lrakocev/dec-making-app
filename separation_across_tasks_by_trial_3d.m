function separation_across_tasks_by_trial_3d(tasks, tables, colors, save_to)
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

    dec_eng = get_arr(table.eng_around_dec);
    mean_dec_eng = mean(dec_eng);
    std_err_dec_eng = std(dec_eng) / length(dec_eng);

    h = scatter3(mean_timing, mean_eng, mean_dec_eng, 20, colors(s), 'filled');
    hold on 
    plot3([mean_timing, mean_timing]', [-std_err_eng, std_err_eng]'+mean_eng', [mean_dec_eng, mean_dec_eng]', colors(s))
    hold on 
    plot3([-std_err_timing, std_err_timing]'+mean_timing', [mean_eng, mean_eng]',[mean_dec_eng, mean_dec_eng]', colors(s))
    hold on
    plot3([mean_timing, mean_timing]', [mean_eng, mean_eng]',[-std_err_dec_eng, std_err_dec_eng]'+mean_dec_eng', colors(s))
    hold on

    hs = [hs; h];
end

title("3D Separation of Eyetracking Data By Trial")
legend(hs, legend_entry);
xlabel('Timing')
ylabel('Avg Eng')
zlabel('Eng Around Decision')
fighandle = gcf;
set(gcf,'renderer','Painters')
saveas(fighandle,strcat(save_to,'3d_plot_by_trials.fig'))


end