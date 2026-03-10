%% =========================================================
%  CEN 575 / ECE 468 - Assignment 4: Structure from Motion
%  main_sfm.m  —  Complete pipeline
%  MATLAB R2024a | Samsung Galaxy A12
%
%  Folder structure expected:
%    Code/        ← run from here
%    Data/        ← img.jpeg, img1.jpeg ... img41.jpeg
%    Results/     ← auto-created
%% =========================================================
clc; clear; close all;

%% ── 0. Configuration ─────────────────────────────────────
cfg.dataDir        = fullfile('..', 'Data');
cfg.resultsDir     = fullfile('..', 'Results');
cfg.matchRatio     = 0.75;    % Lowe ratio test threshold
cfg.ransacThresh   = 1.5;     % pixels — RANSAC epipolar distance
cfg.minInliers     = 10;      % minimum inliers to accept a pair
cfg.numNeighbours  = 5;       % each image matched to N neighbours each side
cfg.maxDepth       = 500;     % discard triangulated points beyond this depth
cfg.denoiseK       = 10;      % pcdenoise num neighbours
cfg.denoiseThresh  = 1.5;     % pcdenoise sigma multiplier

% Samsung Galaxy A12 — main camera (48MP sensor, 3.9mm focal length)
cfg.focalLength_mm = 3.9;
cfg.sensorWidth_mm = 5.16;    % from sensor area ~20mm^2

if ~exist(cfg.resultsDir, 'dir'), mkdir(cfg.resultsDir); end
fprintf('=== Structure from Motion Pipeline (R2024a) ===\n\n');

%% ── 1. Load Images ───────────────────────────────────────
fprintf('[1] Loading images from: %s\n', cfg.dataDir);

imgFiles = dir(fullfile(cfg.dataDir, 'img*.jpeg'));
assert(~isempty(imgFiles), 'No images found in Data folder.');

% Natural sort: img.jpeg (-1), img1.jpeg (1), img2.jpeg (2), ...
numericPart = zeros(1, numel(imgFiles));
for i = 1:numel(imgFiles)
    tok = regexp(imgFiles(i).name, 'img(\d*)\.jpeg', 'tokens');
    s   = tok{1}{1};
    numericPart(i) = ifelse(isempty(s), -1, str2double(s));
end
[~, sidx] = sort(numericPart);
imgFiles  = imgFiles(sidx);
N         = numel(imgFiles);
imgPaths  = fullfile(cfg.dataDir, {imgFiles.name});
fprintf('    Found %d images.\n\n', N);

%% ── 2. Build Camera Intrinsics ───────────────────────────
fprintf('[2] Building camera intrinsics...\n');
img1       = imread(imgPaths{1});
[H, W, ~]  = size(img1);
fx = cfg.focalLength_mm / cfg.sensorWidth_mm * W;
fy = fx;
cx = W / 2;
cy = H / 2;
camParams = cameraIntrinsics([fx fy], [cx cy], [H W]);
fprintf('    Image size : %d x %d px\n', W, H);
fprintf('    fx = fy    : %.2f px\n', fx);
fprintf('    Principal  : (%.1f, %.1f)\n\n', cx, cy);

%% ── 3. Feature Extraction (SIFT) ─────────────────────────
fprintf('[3] Extracting SIFT features...\n');
feats  = cell(1, N);
kpts   = cell(1, N);

for i = 1:N
    img  = imread(imgPaths{i});
    gray = im2gray(img);
    pts  = detectSIFTFeatures(gray, ...
               'NumLayersInOctave', 3, ...
               'ContrastThreshold', 0.01, ...
               'EdgeThreshold',     10);
    [desc, pts] = extractFeatures(gray, pts, 'Method', 'SIFT');
    feats{i} = desc;
    kpts{i}  = pts;
    fprintf('    Image %02d / %02d : %d keypoints\n', i, N, pts.Count);
end
fprintf('\n');

%% ── 4. Pairwise Feature Matching ─────────────────────────
fprintf('[4] Matching features (%d neighbours each side)...\n', cfg.numNeighbours);

% Build candidate pairs
pairs = [];
for i = 1:N
    for j = i+1 : min(i + cfg.numNeighbours, N)
        pairs = [pairs; i j]; %#ok<AGROW>
    end
end
nPairs = size(pairs, 1);
fprintf('    Candidate pairs: %d\n', nPairs);

% Pre-allocate
inlPts1  = cell(nPairs, 1);
inlPts2  = cell(nPairs, 1);
validPair = false(nPairs, 1);

for k = 1:nPairs
    i = pairs(k,1);
    j = pairs(k,2);

    idxPairs = matchFeatures(feats{i}, feats{j}, ...
        'MatchThreshold', 100, ...
        'MaxRatio',       cfg.matchRatio, ...
        'Unique',         true);

    if size(idxPairs, 1) < cfg.minInliers, continue; end

    p1 = kpts{i}(idxPairs(:,1));
    p2 = kpts{j}(idxPairs(:,2));

    try
        [~, inliers] = estimateFundamentalMatrix( ...
            p1.Location, p2.Location, ...
            'Method',            'RANSAC', ...
            'NumTrials',         2000, ...
            'DistanceThreshold', cfg.ransacThresh, ...
            'Confidence',        99);

        if sum(inliers) < cfg.minInliers, continue; end

        inlPts1{k}  = p1(inliers);
        inlPts2{k}  = p2(inliers);
        validPair(k) = true;
        fprintf('    Pair %02d-%02d : %d inliers\n', i, j, sum(inliers));
    catch
        % skip geometrically degenerate pairs
    end
end

validIdx = find(validPair);
fprintf('    Valid pairs: %d / %d\n\n', numel(validIdx), nPairs);
assert(numel(validIdx) > 0, 'No valid pairs found. Check image overlap.');

%% ── 5. Incremental Pose Estimation ───────────────────────
fprintf('[5] Estimating camera poses...\n');

% Seed = valid pair with most inliers
inlierCounts = cellfun(@(x) size(x,1), inlPts1(validPair));
[~, bestLoc] = max(inlierCounts);
seedK = validIdx(bestLoc);
seedI = pairs(seedK, 1);
seedJ = pairs(seedK, 2);
fprintf('    Seed pair  : images %d & %d (%d inliers)\n', ...
    seedI, seedJ, inlierCounts(bestLoc));

% Initialise pose structs
camPoses = struct( ...
    'R',          repmat({eye(3)},    N, 1), ...
    'Translation',repmat({[0 0 0]},   N, 1), ...
    'registered', repmat({false},     N, 1));

% Register seed image I at world origin
camPoses(seedI).registered = true;

% Register seed image J relative to I
E_seed = estimateEssentialMatrix( ...
    inlPts1{seedK}.Location, inlPts2{seedK}.Location, camParams);
[R_rel, t_rel] = relativeCameraPose(E_seed, camParams, ...
    inlPts1{seedK}.Location, inlPts2{seedK}.Location);
camPoses(seedJ).R           = R_rel;
camPoses(seedJ).Translation = t_rel(:)';
camPoses(seedJ).registered  = true;
fprintf('    Camera %02d registered (seed).\n', seedJ);

% Propagate poses iteratively
changed = true;
while changed
    changed = false;
    for k = 1:nPairs
        if ~validPair(k), continue; end
        i = pairs(k,1);  j = pairs(k,2);

        % Forward: i registered → register j
        if camPoses(i).registered && ~camPoses(j).registered
            try
                E = estimateEssentialMatrix( ...
                    inlPts1{k}.Location, inlPts2{k}.Location, camParams);
                [R_rel, t_rel] = relativeCameraPose(E, camParams, ...
                    inlPts1{k}.Location, inlPts2{k}.Location);
                camPoses(j).R           = camPoses(i).R * R_rel;
                camPoses(j).Translation = camPoses(i).Translation(:)' + ...
                                          (camPoses(i).R * t_rel(:))';
                camPoses(j).registered  = true;
                changed = true;
                fprintf('    Camera %02d registered.\n', j);
            catch; end

        % Backward: j registered → register i
        elseif camPoses(j).registered && ~camPoses(i).registered
            try
                E = estimateEssentialMatrix( ...
                    inlPts2{k}.Location, inlPts1{k}.Location, camParams);
                [R_rel, t_rel] = relativeCameraPose(E, camParams, ...
                    inlPts2{k}.Location, inlPts1{k}.Location);
                camPoses(i).R           = camPoses(j).R * R_rel;
                camPoses(i).Translation = camPoses(j).Translation(:)' + ...
                                          (camPoses(j).R * t_rel(:))';
                camPoses(i).registered  = true;
                changed = true;
                fprintf('    Camera %02d registered.\n', i);
            catch; end
        end
    end
end

nReg = sum([camPoses.registered]);
fprintf('    Registered : %d / %d cameras.\n\n', nReg, N);

%% ── 6. Triangulation ─────────────────────────────────────
fprintf('[6] Triangulating 3D points...\n');
xyzAll    = [];
colorsAll = [];

for k = 1:nPairs
    if ~validPair(k), continue; end
    i = pairs(k,1);  j = pairs(k,2);
    if ~camPoses(i).registered || ~camPoses(j).registered, continue; end

    % Build projection matrices  [R|t] composed with K
    P1 = buildProjection(camParams, camPoses(i));
    P2 = buildProjection(camParams, camPoses(j));

    pts3D = triangulate( ...
        inlPts1{k}.Location, inlPts2{k}.Location, P1, P2);

    % Depth filter — discard points behind camera or implausibly far
    valid = pts3D(:,3) > 0.01 & pts3D(:,3) < cfg.maxDepth;
    pts3D = pts3D(valid, :);
    if isempty(pts3D), continue; end

    % Per-pair MAD outlier removal (removes exploding triangulations)
    med = median(pts3D);
    sd  = 1.4826 * median(abs(pts3D - med));
    sd(sd < 1e-6) = 1e-6;
    inBounds = all(abs(pts3D - med) < 4 * sd, 2);
    pts3D = pts3D(inBounds, :);
    if isempty(pts3D), continue; end
    % Sync valid index for colour sampling
    tmp = find(valid); valid(:) = false; valid(tmp(inBounds)) = true;

    % Sample colours from image i
    img  = imread(imgPaths{i});
    loc  = round(inlPts1{k}.Location(valid, :));
    loc(:,1) = min(max(loc(:,1), 1), W);
    loc(:,2) = min(max(loc(:,2), 1), H);
    rgb  = zeros(size(pts3D,1), 3, 'uint8');
    for m = 1:size(loc,1)
        rgb(m,:) = squeeze(img(loc(m,2), loc(m,1), :))';
    end

    xyzAll    = [xyzAll;    pts3D]; %#ok<AGROW>
    colorsAll = [colorsAll; rgb];   %#ok<AGROW>
end
fprintf('    3D points before denoising: %d\n\n', size(xyzAll,1));

%% ── 7. Statistical Outlier Removal ───────────────────────
fprintf('[7] Removing outliers...\n');
ptCloud = pointCloud(xyzAll, 'Color', colorsAll);
if ptCloud.Count > 50
    ptCloud = pcdenoise(ptCloud, ...
        'NumNeighbors', cfg.denoiseK, ...
        'Threshold',    cfg.denoiseThresh);
    % Second global MAD pass to remove any remaining stragglers
    xyz = ptCloud.Location;
    col = ptCloud.Color;
    med = median(xyz);
    sd  = 1.4826 * median(abs(xyz - med));
    sd(sd < 1e-6) = 1e-6;
    keep = all(abs(xyz - med) < 5 * sd, 2);
    ptCloud = pointCloud(xyz(keep,:), 'Color', col(keep,:));
    fprintf('    Points after denoising: %d\n\n', ptCloud.Count);
else
    fprintf('    Too few points — skipping denoising.\n\n');
end

%% ── 8. Visualization ─────────────────────────────────────
fprintf('[8] Visualizing...\n');

% ---- 8a. Point Cloud ----
figure('Name','3D Point Cloud','Color','k','Position',[50 50 960 720]);
pcshow(ptCloud, 'MarkerSize', 30);
title('Reconstructed 3D Point Cloud', 'Color','w', 'FontSize',14);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
saveas(gcf, fullfile(cfg.resultsDir, 'pointcloud_view.png'));
fprintf('    Saved: pointcloud_view.png\n');

% ---- 8b. Camera Trajectory ----
figure('Name','Camera Trajectory','Color','w','Position',[100 50 960 720]);
hold on; grid on; axis equal;
cmap     = jet(N);
centres  = nan(N, 3);

for i = 1:N
    if ~camPoses(i).registered, continue; end
    t = camPoses(i).Translation(:)';
    R = camPoses(i).R;
    plotCamera('Location', t, 'Orientation', R, ...
               'Size', 0.05, 'Color', cmap(i,:), 'Label', num2str(i));
    centres(i,:) = t;
end

valid_c = ~any(isnan(centres), 2);
if sum(valid_c) > 1
    plot3(centres(valid_c,1), centres(valid_c,2), centres(valid_c,3), ...
          'k--', 'LineWidth', 1.5);
end
title(sprintf('Camera Trajectory  (%d / %d registered)', nReg, N), 'FontSize',14);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
saveas(gcf, fullfile(cfg.resultsDir, 'camera_trajectory.png'));
fprintf('    Saved: camera_trajectory.png\n');

% ---- 8c. Registered Views Bar ----
figure('Name','Registered Views','Color','w');
regVec = double([camPoses.registered]);
bar(1:N, regVec, 'FaceColor',[0.2 0.55 0.9], 'EdgeColor','none');
xlabel('Camera Index'); ylabel('Registered');
title(sprintf('Registered Cameras: %d / %d', nReg, N), 'FontSize',13);
yticks([0 1]); yticklabels({'No','Yes'}); ylim([0 1.3]);
saveas(gcf, fullfile(cfg.resultsDir, 'registered_views.png'));
fprintf('    Saved: registered_views.png\n\n');

%% ── 9. Save Results ──────────────────────────────────────
fprintf('[9] Saving results...\n');

plyFile = fullfile(cfg.resultsDir, 'reconstruction.ply');
pcwrite(ptCloud, plyFile, 'Encoding', 'ascii');
fprintf('    Saved PLY : %s  (%d points)\n', plyFile, ptCloud.Count);

save(fullfile(cfg.resultsDir, 'sfm_results.mat'), ...
    'xyzAll', 'colorsAll', 'camPoses', 'camParams', 'imgPaths', 'pairs');
fprintf('    Saved MAT : sfm_results.mat\n\n');

fprintf('=== SfM Pipeline Complete ===\n');
fprintf('    Registered cameras : %d / %d\n', nReg, N);
fprintf('    3D points          : %d\n', ptCloud.Count);
fprintf('    Results saved to   : %s\n', cfg.resultsDir);

%% =========================================================
%  LOCAL HELPER FUNCTIONS
%% =========================================================

function P = buildProjection(camParams, pose)
%BUILDPROJECTION  Build a 4x3 camera matrix for MATLAB's triangulate().
%
%   MATLAB triangulate() convention:  x = X * P
%     X : 1x4 homogeneous world point (row vector)
%     P : 4x3  =  [R; t] * K
%
%   relativeCameraPose stores Translation as the camera CENTRE in world
%   coordinates (like a Location).  We must convert to the
%   world-to-camera translation:   t_wc = -R * C
%
    K = [camParams.FocalLength(1), 0,                         camParams.PrincipalPoint(1);
         0,                        camParams.FocalLength(2),   camParams.PrincipalPoint(2);
         0,                        0,                          1];

    R = pose.R;                  % 3x3  world-to-camera rotation
    C = pose.Translation(:)';    % 1x3  camera centre in world coords
    t = -C * R';                 % 1x3  world-to-camera translation (t = -R*C)

    P = [R; t] * K;              % 4x3
end

function result = ifelse(cond, a, b)
%IFELSE  Inline ternary helper.
    if cond, result = a; else, result = b; end
end