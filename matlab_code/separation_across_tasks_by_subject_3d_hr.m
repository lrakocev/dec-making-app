function separation_across_tasks_by_subject_3d_hr(tasks,tables,colors,save_to)
hs = [];
figure
legend_entry = [];
timings = [];
engs = [];
for i = 1:length(tasks)
    tit = tasks(i);
    table = tables(i);
    table = table{1};
    table = table(str2double(table.number_of_timesteps(:)) > 9, :);

    subjects = unique(table.subjectidnumber);
    legend_entry = [legend_entry; tit + ": n = " + string(length(subjects))];

    curr_mean_min_percent = [];
    curr_mean_max_percent = [];
    curr_mean_direction = [];

    for s = 1:length(subjects)
        subid = subjects(s);
        curr_table = table(table.subjectidnumber == string(subid{1}), :);

        min_percent = get_arr(curr_table.percent_min_hr); 
        mean_min_percent = mean(min_percent, 'omitnan');
        curr_mean_min_percent = [curr_mean_min_percent; mean_min_percent];
    
        max_percent = get_arr(curr_table.percent_max_hr);
        mean_max_percent = mean(max_percent, 'omitnan');
        curr_mean_max_percent = [curr_mean_max_percent; mean_max_percent];

        direction = get_arr(curr_table.direction);
        mean_direction = mean(direction, 'omitnan');
        curr_mean_direction = [curr_mean_direction; mean_direction];
        
    end

    mean_min_percent = mean(curr_mean_min_percent, 'omitnan');
    mean_max_percent = mean(curr_mean_max_percent, 'omitnan');
    mean_direction = mean(curr_mean_direction, 'omitnan');
    std_err_timing = std(curr_mean_min_percent, 'omitnan') / sum(~isnan(curr_mean_min_percent));
    std_err_eng = std(curr_mean_max_percent, 'omitnan') / sum(~isnan(curr_mean_max_percent));
    std_err_dec_eng = std(curr_mean_direction, 'omitnan') / sum(~isnan(curr_mean_direction));
    
    h = scatter3(mean_min_percent, mean_max_percent, mean_direction, 20, colors(i), 'filled');
    hold on 
    plot3([mean_min_percent, mean_min_percent]', [-std_err_eng, std_err_eng]'+mean_max_percent', [mean_direction, mean_direction]', colors(i))
    hold on 
    plot3([-std_err_timing, std_err_timing]'+mean_min_percent', [mean_max_percent, mean_max_percent]',[mean_direction, mean_direction]', colors(i))
    hold on
    plot3([mean_min_percent, mean_min_percent]', [mean_max_percent, mean_max_percent]',[-std_err_dec_eng, std_err_dec_eng]'+mean_direction', colors(i))
    hold on

    hs = [hs; h];
end
    
title("3D Separation of Eyetracking Data By Subject")
legend(hs, legend_entry);
xlabel('% min below avg')
ylabel('% max above avg')
zlabel('direction')
fighandle = gcf;
set(gcf,'renderer','Painters')
saveas(fighandle,strcat(save_to,'3d_plot_by_subjects.fig'))

end