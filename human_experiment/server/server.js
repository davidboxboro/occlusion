// server.js

    // imports
    import redis from 'redis';
    import express from 'express';
    import fs from 'fs';
    import md5 from 'js-md5';
    import { Mutex } from 'async-mutex';

    // vars
    const redisPort = 6379;
    const dbIndex = 12;
    const serverPort = 8090;
    const expID = "hexp1.1_interp"  // "hexp1.1_ddnm_1000";
    const taskPath = `../client/${expID}/id_to_im_paths.json`;
    const labelMappingPath = "../../data/imagenet/ann/hr_label_mappings.json";
    const clientIconPath = "icons";
    const expiryMin = 60 * 4; // 4 hours;
    const allowedGroupIdxs = [0, 1];

    // create app
    const app = express();

    // connect to redis
    const client = redis.createClient({ port: redisPort });
    client.on("error", (err) => console.log("Redis client error", err));
    await client.connect();
    console.log("Connected to redis");
    client.select(dbIndex);

    // load experiment data at server start
    const taskIDToImgPaths = JSON.parse(fs.readFileSync(taskPath));
    const allTaskIDs = Object.keys(taskIDToImgPaths);
    const taskIDs = allTaskIDs.filter(task => allowedGroupIdxs.includes(parseInt(task.split("-")[0])));
    console.log("Loaded task ID to image path mappings");
    console.log("Num tasks: %s", taskIDs.length);
    console.log("Task IDs: %s", taskIDs);

    // number of trials per user
    const numTrials = taskIDToImgPaths[taskIDs[0]].length;
    // const numTrials = 5;
    console.log("Number of trials: %s", numTrials);

    // mapping of task IDs
    const taskMapName = `${expID}:taskMap`;
    // add task IDs to task map if they don't exist yet
    for (const taskID of taskIDs) {
        if (!await client.hExists(taskMapName, taskID)) {
            console.log("Creating key for task ID %s in %s", taskID, taskMapName);
            await client.hSet(taskMapName, taskID, JSON.stringify([]));
        }
    }

    // set of user IDs seen so far
    const userSetName = `${expID}:userSet`;

    // load labels and icons
    const labelMapping = JSON.parse(fs.readFileSync(labelMappingPath));
    const labels = Object.keys(labelMapping);
    labels.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase())); // sort labels alphabetically
    const iconPaths = labels.map(
        (label) => clientIconPath + "/" + label.split(" ").join("_") + ".png"
    );
    console.log("Loaded labels and icon paths");

    // express parsing middleware
    app.use(express.urlencoded({ extended: true })); // to parse url-encoded data
    app.use(express.json()); // to parse the incoming requests with JSON payloads

    // serve client static page in ../client dir
    app.use(express.static("../client"));

    // called by client to fetch labels and icon paths
    app.get("/get-labels", (_, res) => {
        console.log("\nReceived /get-labels GET request");
        res.status(200).send({
            labels: labels,
            iconPaths: iconPaths,
        });
    });

    // mutex to prevent concurrency issues
    const mutex = new Mutex();

    // get task ID given user ID
    async function getTaskID(userID) {
        // if user has already been assigned a task that hasn't expired yet, then give user that task again
        for (const taskID of taskIDs) {
            const assignedUserVals = JSON.parse(await client.hGet(taskMapName, taskID));
            if (assignedUserVals.length === 0) {
                continue;
            }
            // not expired means that it must be the last user ID in the vals list
            const assignedUserVal = assignedUserVals[assignedUserVals.length - 1];
            const assignedUserID = assignedUserVal.userID;
            const assignedUserAccept = assignedUserVal.accept;
            const assignedUserStartTime = assignedUserVal.startTime;
            const minElapsed = ((new Date()).getTime() - assignedUserStartTime) / (1000 * 60);
            // accept == false means that user's task has expired
            if (userID === assignedUserID && assignedUserAccept === true && minElapsed <= expiryMin) {
                const lastTrialIdx = assignedUserVal.lastTrialIdx;
                console.log("\tAssigning task ID %s to user ID %s (same user ID previously assigned)", taskID, userID);
                return { taskID: taskID, lastTrialIdx: lastTrialIdx };
            }
        }

        // if user has already been assigned a task in the past, then the user cannot
        // be assigned a new task again
        const assignedUserIDs = await client.sMembers(userSetName);
        console.log("\tUser ID set (before adding current user ID): %s", JSON.stringify(assignedUserIDs, null, 4).replace(/\n/g, "\n\t"));
    if (assignedUserIDs.includes(userID)) {
        console.log("\tUser ID %s seen before and previous task has expired, not assigning any task", userID);
        return { taskID: null, lastTrialIdx: null };
    }

    // otherwise, assign user a task that either (1) hasn't been assigned yet
    // or (2) was assigned to a previous user but their time has expired
    // and they have not finished their trials yet.
    // currently, trials that have been partially completed by a user but not
    // finished do not get reassigned to anyone.
    // IMPORTANT: hold lock while running to prevent two users from picking same task
    const newTask = await mutex.runExclusive(async () => {
        console.log("\tAcquiring lock");
        for (const taskID of taskIDs) {
            const assignedUserVals = JSON.parse(await client.hGet(taskMapName, taskID));
            let assignedUserID;
            let assignedUserLastTrialIdx;
            let assignedUserStartTime;
            let assignedUserAccept;
            let minElapsed;
            // if a user has been assigned this task before, get their user ID and other info
            if (assignedUserVals.length !== 0) {
                const assignedUserVal = assignedUserVals[assignedUserVals.length - 1];
                assignedUserID = assignedUserVal.userID;
                assignedUserLastTrialIdx = assignedUserVal.lastTrialIdx;
                assignedUserStartTime = assignedUserVal.startTime;
                assignedUserAccept = assignedUserVal.accept;
                // get time elapsed since user started task
                const curTime = (new Date()).getTime();
                minElapsed = (curTime - assignedUserStartTime) / (1000 * 60);
                console.log("\tUser ID %s has completed %s of %s trials, with %s minutes passed since start time",
                    assignedUserID, assignedUserLastTrialIdx + 1, numTrials, minElapsed
                );
            }
            // assign task to new user if no old user exists, or a lot of time passed since old user started task
            if ((assignedUserID === undefined) || (assignedUserAccept === false)) {
                if (assignedUserID === undefined) {
                    console.log("\tAssigning task ID %s to user ID %s (no user ID previously assigned)",
                        taskID, userID
                    );
                } else {
                    console.log("\tAssigning task ID %s to user ID %s (user ID %s previously assigned but responses not accepted)",
                        taskID, userID, assignedUserID
                    );
                }
                // assign user that task
                const initTrialIdx = -1;
                const startTime = (new Date()).getTime();
                const userVal = {
                    userID: userID,
                    lastTrialIdx: initTrialIdx,
                    startTime: startTime,
                    endTime: startTime,
                    accept: true, // default option is to accept
                };
                assignedUserVals.push(userVal);
                await client.hSet(taskMapName, taskID, JSON.stringify(assignedUserVals));
                // add user to user set, so that it cannot be assigned another task in the future
                await client.sAdd(userSetName, userID);
                console.log("\tReleasing lock");
                return { taskID: taskID, lastTrialIdx: initTrialIdx };
            }
        }
        console.log("\tReleasing lock");
        return { taskID: null, lastTrialIdx: null };
    });
    // return new task (in mutex)
    return newTask;
}

// called by client at the beginning of a task
// returns full set of image paths corresponding to task ID
app.post("/get-img-paths", async (req, res) => {
    console.log("\nReceived /get-img-paths POST request with args %s", JSON.stringify(req.body, null, 4));
    const userID = req.body.userID;
    const { taskID, lastTrialIdx } = await getTaskID(userID);
    if (taskID !== null) {
        console.log("\tTask ID %s (with last trial index %s) given to user ID %s", taskID, lastTrialIdx, userID);
        const imgPaths = taskIDToImgPaths[taskID].map(d => d['image']).slice(0, numTrials);
        res.status(200).send({
            imgPaths: imgPaths,
            taskID: taskID,
            lastTrialIdx: lastTrialIdx,
        });
    } else {
        console.log("\tWARNING: No task ID given to user ID %s", userID);
        res.status(400).end();
    }
});

// called by client after submitting each trial
// server stores trial response in redis
app.post("/submit-response", async (req, res) => {
    const userID = req.body.userID;
    const taskID = req.body.taskID;
    const imgPath = req.body.imgPath;
    const selLabel = req.body.selLabel;
    const trialIdx = req.body.trialIdx;
    const time = (new Date()).getTime();
    console.log("\nReceived /submit-response POST request with args %s",
        JSON.stringify(req.body, null, 4)
    );
    // update task map
    const assignedUserVals = JSON.parse(await client.hGet(taskMapName, taskID));
    const val = assignedUserVals[assignedUserVals.length - 1];
    console.log("\tRead task map: task ID %s has last value %s", taskID, JSON.stringify(val, null, 4).replace(/\n/g, "\n\t"));
    const savedUserID = val.userID;
    const savedLastTrialIdx = val.lastTrialIdx;
    const savedEndTime = val.endTime;
    if (savedUserID !== userID) {
        console.log("\tERROR: Saved user ID was %s, but user ID is %s", savedUserID, userID);
        res.status(400).end();
    } else if (savedLastTrialIdx !== trialIdx - 1) {
        console.log("\tERROR: Saved last trial index was %s, but trial index is %s", savedLastTrialIdx, trialIdx);
        res.status(400).end();
    } else if (savedEndTime > time) {
        console.log("\tERROR: Saved time (%s) bigger than current time (%s)", savedEndTime, time);
        res.status(400).end();
    } else {
        val.lastTrialIdx = trialIdx;
        val.endTime = time;
        console.log("\tCurrent time is %s seconds after the last submission", (time - savedEndTime) / 1000);
        assignedUserVals[assignedUserVals.length - 1] = val;
        await client.hSet(taskMapName, taskID, JSON.stringify(assignedUserVals));
        console.log("\tUpdated task map: task ID %s has last value %s", taskID, JSON.stringify(val, null, 4).replace(/\n/g, "\n\t"));
        // save user's response
        const responseMapName = `${expID}:${taskID}:${userID}`;
        await client.hSet(responseMapName, imgPath, selLabel);
        console.log("\tUpdated response map %s: image path %s has value %s", responseMapName, imgPath, selLabel);
        res.status(200).end();
    }
});

// called by client to hash a number
app.post("/get-hash", (req, res) => {
    console.log("\nReceived get-hash POST request");
    res.status(200).send({
        hash: md5(req.body.val),
    });
});

// app listens to port for requests
// call at end, so that routes and middleware are set up properly first
app.listen(serverPort);
