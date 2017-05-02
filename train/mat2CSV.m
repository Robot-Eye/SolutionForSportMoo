clc;
clear all;
CSV_file = load('D:\TsingHua_University\Êµ¼ù&Éç¹¤\visual_solver\mat2CSV\mpii_human_pose_v1_u12_1.mat');
img_num = 24987;

ff = [];
IMAGE = cell(1,1);

nn = {'img_id','img_name','scale','head rectangle_x1','head rectangle_y1','head rectangle_x2','head rectangle_y2','human_position_x','human_position_y','r_ankle_x','r_ankle_y','is_visible','r_knee_x','r_knee_y','is_visible','r_hip_x','r_hip_y','is_visible','l_hip_x','l_hip_y','is_visible','l_knee_x','l_knee_y','is_visible','l_ankle_x','l_ankle_y','is_visible','pelvis_x','pelvis_y','is_visible','thorax_x','thorax_y','is_visible','upper neck_x','upper neck_y','is_visible','head top_x','head top_y','is_visible','r wrist_x','r wrist_y','is_visible','r elbow_x','r elbow_y','is_visible','r shoulder_x','r shoulder_y','is_visible','l shoulder_x','l shoulder_y','is_visible','l elbow_x','l elbow_y','is_visible','l wrist_x','l wrist_y','is_visible'};

for i = 1:length(nn)
    IMAGE{1,i} = nn{i};
end
count = 1;
for id = 1:img_num
    if mod(id,100)==0
        fprintf('Processing: %d >> %f \n',id,id*100/img_num);
    end
%     fprintf(num2str(id));
%     fprintf('\n');
    im_name = CSV_file.RELEASE.annolist(id).image.name;
    im_annorect = CSV_file.RELEASE.annolist(id).annorect;
    im_annorect_num = length(im_annorect);
    for ii = 1:im_annorect_num
        IMAGE{count+ii,1} = id;
        IMAGE{count+ii,2} = im_name;
        if (isfield(im_annorect,'scale'))
            IMAGE{count+ii,3} = CSV_file.RELEASE.annolist(id).annorect(ii).scale;
        end
        if (isfield(im_annorect,'x1'))
            IMAGE{count+ii,4} = CSV_file.RELEASE.annolist(id).annorect(ii).x1;
        end
        if (isfield(im_annorect,'y1'))
            IMAGE{count+ii,5} = CSV_file.RELEASE.annolist(id).annorect(ii).y1;
        end  
        if (isfield(im_annorect,'x2'))
            IMAGE{count+ii,6} = CSV_file.RELEASE.annolist(id).annorect(ii).x2;
        end        
        if (isfield(im_annorect,'y2'))
            IMAGE{count+ii,7} = CSV_file.RELEASE.annolist(id).annorect(ii).y2;
        end        
        if (isfield(im_annorect,'objpos'))
            if (isfield(im_annorect(ii).objpos,'x'))
                IMAGE{count+ii,8} = CSV_file.RELEASE.annolist(id).annorect(ii).objpos.x;
                left_up_x = IMAGE{count+ii,8} - 100 * IMAGE{count+ii,3};
                right_down_x = IMAGE{count+ii,8} + 100 * IMAGE{count+ii,3};
            end
            if (isfield(im_annorect(ii).objpos,'y'))
                IMAGE{count+ii,9} = CSV_file.RELEASE.annolist(id).annorect(ii).objpos.y;
                left_up_y = IMAGE{count+ii,9} - 100 * IMAGE{count+ii,3};
                right_down_y = IMAGE{count+ii,9} + 100 * IMAGE{count+ii,3};
            end            
        end
        if (isfield(im_annorect,'annopoints'))
            if (isfield(im_annorect(ii).annopoints,'point'))
                point_num = length(CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point);
                for jj = 1:point_num
                    point_id = CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point(jj).id + 1;
                    IMAGE{count+ii,9+3*point_id-2} = CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point(jj).x - left_up_x;
                    IMAGE{count+ii,9+3*point_id-1} = CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point(jj).y - left_up_y;
                    if (isfield(im_annorect(ii).annopoints.point(jj),'is_visible'))
                        IMAGE{count+ii,9+3*point_id} = double(CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point(jj).is_visible);                        
                        if (ischar(CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point(jj).is_visible))
                           if (CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point(jj).is_visible == '1')
                               IMAGE{count+ii,9+3*point_id} = double(1);
                           end
                           if (CSV_file.RELEASE.annolist(id).annorect(ii).annopoints.point(jj).is_visible == '0')
                               IMAGE{count+ii,9+3*point_id} = double(0);
                           end
                        end
                    end
                end
            end
        end         
        
    end
    count = count + im_annorect_num;
end
% IMAGE = load('IMAGE.mat');
% IMAGE = IMAGE.IMAGE;
 save('IMAGE.mat','IMAGE');
cell2csv('data.csv',IMAGE);
