function separation_across_tasks_by_subject(tasks,tables,colors,save_to)
hs = [];
figure
legend_entry = [];
timings = [];
engs = [];
for i = 1:length(tasks)
    tit = tasks(i);
    table = tables(i);
    table = table{1};

    subjects = unique(table.subjectidnumber);
    legend_entry = [legend_entry; tit + ": n = " + string(length(subjects))];


    curr_mean_timing = [];
    curr_mean_eng = [];

    for s = 1:length(subjects)
        subid = subjects(s);
        curr_table = table(table.subjectidnumber == string(subid{1}), :);

        curr_table.decision_timing = get_arr(curr_table.decision_timing);
        curr_table = curr_table(curr_table.decision_timing ~= 0, :);
        
        timing = curr_table.decision_timing; 
        mean_timing = mean(timing, 'omitnan');
        curr_mean_timing = [curr_mean_timing; mean_timing];
    
        eng = get_arr(curr_table.avg_eng);
        mean_eng = mean(eng, 'omitnan');
        curr_mean_eng = [curr_mean_eng; mean_eng];
        
    end

    mean_timing = mean(curr_mean_timing, 'omitnan');
    mean_eng = mean(curr_mean_eng, 'omitnan');
    std_err_timing = std(curr_mean_timing, 'omitnan') / sum(~isnan(curr_mean_timing));
    std_err_eng = std(curr_mean_eng, 'omitnan') / sum(~isnan(curr_mean_eng));
    

    h = scatter(mean_timing, mean_eng, 20, colors(i), 'filled');
    hold on 
    plot([mean_timing, mean_timing]', [-std_err_eng, std_err_eng]'+mean_eng', colors(i))
    hold on 
    plot([-std_err_timing, std_err_timing]'+mean_timing', [mean_eng, mean_eng]', colors(i))
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
    

legend(hs, legend_entry)
title('Timing vs Avg Eng by Subject')
xlabel('Timing')
ylabel('Avg Eng')
fighandle = gcf;
set(gcf,'renderer','Painters')
saveas(fighandle,strcat(save_to,'2d_plot_by_subjects.fig'))

end