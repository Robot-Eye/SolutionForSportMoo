require('nn')
require('cunn')
require('cutorch')
require('image')
require('nngraph')
require('sys')

DATA_DIR = './data/'
RESULT_DIR = './result/'
VIDEO_NAME = 'IMG_1775_A'
VIDEO_PATH = DATA_DIR..VIDEO_NAME

TPM_pose_estimation = torch.load('./model/TPM_pose_estimation_lite.t7')
TPM_pose_estimation:cuda()
-- Modify Here to change data dir:
data_dir = VIDEO_PATH..'/'


require'lfs'
require('torchx')
cv = require('cv')
require('cv.imgproc')
require('cv.highgui')

file_output = assert(io.open(RESULT_DIR..VIDEO_NAME..'/golf_pose.csv','w'))
file_output:write('file_name,head_x,head_y,head_confidence,neck_x,neck_y,neck_confidence,Rshoulder_x,Rshoulder_y,Rshoulder_confidence,Relbow_x,Relbow_y,Relbow_confidence,Rwrist_x,Rwrist_y,Rwrist_confidence,Lshoulder_x,Lshoulder_y,Lshoulder_confidence,Lelbow_x,Lelbow_y,Lelbow_confidence,Lwrist_x,Lwrist_y,Lwrist_confidence,Rhip_x,Rhip_y,Rhip_confidence,Rknee_x,Rknee_y,Rknee_confidence,Rankle_x,Rankle_y,Rankle_confidence,Lhip_x,Lhip_y,Lhip_confidence,Lknee_x,Lknee_y,Lknee_confidence,Lankle_x,Lankle_y,Lankle_confidence\n')
counter = 0

for file in lfs.dir(data_dir) do
  absolute_file = data_dir..file
    if lfs.attributes(absolute_file,"mode") == "file" and string.find(file, 'png') ~= nil then
      print("running image: "..absolute_file)

      RGB_image = image.load(absolute_file,3,'byte')

      -- hardcoded crop region
      RGB_image = image.crop(RGB_image,124,304,393,875) 
	
      --print("input width: "..RGB_image:size()[2])
     -- print("input heigh: "..RGB_image:size()[3])

      test_image = RGB_image:clone()
      -- RGB to BGR
      test_image[{1,{},{}}] = RGB_image[{3,{},{}}]
      test_image[{3,{},{}}] = RGB_image[{1,{},{}}]

     boxsize = 368  -- we have to estimate the scale of person in the image, or we do multiple scale testing
      npart = 14
      local nchan, height, width = test_image:size(1), test_image:size(2), test_image:size(3)
      print('size: '..height)
      print('size: '..width)

      scale = boxsize/(height)

      imageToTest = image.scale(test_image, width * scale, height * scale, 'bicubic')
      imageToTest = imageToTest:double()
      imageToTest = imageToTest/255 - 0.5

      util = require('tpm_util')
      imageToTest, padding = util.padRightDownCorner(imageToTest)
    --  person_map_resized = image.scale(person_map, person_map:size()[3] * 8, person_map:size()[2] * 8, 'bicubic')

      persons = {{imageToTest:size()[2]/2, imageToTest:size()[3]/2}}

      num_people = table.getn(persons)
      boxsize = 368 -- from configuration file
      person_image = torch.Tensor(3, boxsize, boxsize, num_people):zero()
      for p = 1, num_people do
          for i = 1, boxsize do
              for j = 1, boxsize do
                  x_i = i - boxsize/2 + persons[p][1]
                  y_j = j - boxsize/2 + persons[p][2]
                  if x_i > 0 and x_i <= imageToTest:size()[2] and y_j > 0 and y_j <= imageToTest:size()[3] then
                      person_image[{{},{i},{j},{p}}]= imageToTest[{{},{x_i},{y_j}}]
                  end
              end
          end
      end

      gaussian_map = torch.Tensor(1, boxsize, boxsize):zero()
      sigma = 21.0 -- from configuration file
      for x_p = 1, boxsize do
          for y_p = 1, boxsize do
              dist_sq = (x_p - boxsize/2) * (x_p - boxsize/2) + 
              (y_p - boxsize/2) * (y_p - boxsize/2)
              exponent = dist_sq / 2.0 / sigma / sigma
              gaussian_map[1][x_p][y_p] = torch.exp(-exponent)
          end
      end

      output_blobs_array = {}
      for p = 1, num_people do
          input_4ch = {}
          input_4ch[1] = person_image:select(4,p):cuda()
          input_4ch[2] = gaussian_map:cuda()
          sys.tic()
          output_blobs_array[p] = TPM_pose_estimation:forward(input_4ch):clone():double()
          t = sys.toc()
          print('person '..p..' poes is done, takes about'..t..' s')
      --     itorch.image(  output_blobs_array[p]:select(1,11))
      end


      -- for p = 1, num_people do
      --    print('Person: '..p)
      --    canvas = {}
      --    down_scale_image = image.scale(person_image:select(4,p), person_image:size()[3]/2, person_image:size()[2]/2, 'bicubic')
      --    for _,i in pairs({1,4,8,11,13}) do -- sample 5 body parts:  [head, right elbow, left wrist, right ankle, left knee]
      --        part_map = output_blobs_array[p]:select(1,i)
      --        part_map_resized = image.scale(part_map, part_map:size()[2]*4, part_map:size()[1]*4, 'bicubic')
      --        part_map_color = util.colorize(torch.reshape(part_map_resized, 1, part_map_resized:size()[2], part_map_resized:size()[1]))
      --        table.insert(canvas, part_map_color + down_scale_image*256)
      --    end
      -- end

      limbs = {{ 1 , 2},
       { 3 , 4},
       { 4 , 5},
       { 6 , 7},
       { 7 , 8},
       { 9 , 10},
       {10 , 11},
       {12 , 13},
       {13 , 14}}
      colors = {{0, 0, 255}, {0, 170, 255}, {0, 255, 170}, {0, 255, 0}, {170, 255, 0},
      {255, 170, 0}, {255, 0, 0}, {255, 0, 170}, {170, 0, 255}} -- note BGR ...
      canvas = imageToTest:clone()

      prediction = torch.Tensor(14,3,num_people)
      for p = 1, num_people do
          for i = 1,14 do
               part_map = output_blobs_array[p]:select(1,i)
               part_map_resized = image.scale(part_map, part_map:size()[2]*8, part_map:size()[1]*8, 'bicubic')
               index = unpack(torch.find(part_map_resized, torch.max(part_map_resized)))
               -- print('confidence: '..tostring(torch.max(part_map_resized)))
               --print('index: '..index)
               prediction[i][1][p] = torch.floor(index/(part_map_resized:size()[2])) -- row index
               --print('row: '..prediction[i][1][p])
               prediction[i][2][p] = index%(part_map_resized:size()[2]) -- col index
               --print('col: '..prediction[i][2][p])
               prediction[i][3][p] = torch.max(part_map_resized)
          end
          prediction[{{},{1},{p}}] = prediction[{{},{1},{p}}] - boxsize/2 + persons[p][1]
          prediction[{{},{2},{p}}] = prediction[{{},{2},{p}}] - boxsize/2 + persons[p][2]
      end

      for p = 1, num_people do
        file_output:write(file)
          for i = 1,14 do
              file_output:write(','..tostring(prediction[i][2][p]..','..tostring(prediction[i][1][p])..','..tostring(prediction[i][3][p])))
          end
          file_output:write('\n')
      end
      -- BGR to RGB
      canvas[{1,{},{}}] = imageToTest[{3,{},{}}]
      canvas[{3,{},{}}] = imageToTest[{1,{},{}}]

      canvas = (canvas + 0.5) *255
      canvas = canvas:transpose(1,2):transpose(2,3):clone()
      -- print(canvas:size())

      stickwidth = 6
      for p = 1, num_people do
          for i = 1,14 do
              cv.circle{canvas, {prediction[i][2][p],prediction[i][1][p]}, 3, {255,0,0}, -1}
          end
          for l = 1, 9 do
              cur_canvas = canvas:clone()
              X = prediction[{{limbs[l][1],limbs[l][2]},{1},{p}}]:select(2,1):select(2,1)
              Y = prediction[{{limbs[l][1],limbs[l][2]},{2},{p}}]:select(2,1):select(2,1)
              mX = torch.mean(X)
              mY = torch.mean(Y)
              length = ((X[1]- X[2])^ 2 + (Y[1] - Y[2])^ 2) ^ 0.5
              angle = math.deg(math.atan2(X[1] - X[2], Y[1] - Y[2]))

              cv.ellipse{cur_canvas, {mY,mX},{length/2, stickwidth}, angle, 0, 360, colors[l],-1}
              canvas = canvas * 0.4 + cur_canvas * 0.6 
          end
          
      end
      canvas = canvas:transpose(2,3):transpose(1,2)
      counter = counter + 1
      print("done..."..counter)
      image.save(RESULT_DIR..VIDEO_NAME.."/pm_"..file,canvas/255)
      -- image.save("./cache.jpg",canvas/255)
      collectgarbage() 
      -- break
    end
    
end

