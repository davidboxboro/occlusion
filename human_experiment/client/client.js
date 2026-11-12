// client.js

// states of web app
var states;

// initial states
let stateIdx;
let labelReviewIdx;
let trialIdx;
let lastTrialIdx;

let taskID;
let labels;
let iconPaths;
let imgPaths;
let numTrials;

const numLabelReviewRows = 2;
const numLabelReviewCols = 5;
let numLabelReviewScreens;

const numLabelRows = 7;
const numLabelCols = 8;

let userID;
let studyID;
let sessionID;

let labelReviewClickedState;

const selectColor = "#90EE90";
const baseColor = "white";

const cardWidthInch = 3.37;
let cardWidthPix;

export async function main() {
    // give warning if user tries to reload or exit page
    // TODO: doesn't really work well
    window.addEventListener("beforeunload", () => {
        return true;
    });

    // full screen
    addEventListener("click", () => {
        const el = document.documentElement;
        const rfs = el.requestFullScreen
            || el.webkitRequestFullScreen
            || el.mozRequestFullScreen;
        rfs.call(el);
    });

    // get the query string from the URL
    const queryString = window.location.search;
    console.log("Query string: %s", queryString);

    // create a new URLSearchParams object from the query string
    const urlParams = new URLSearchParams(queryString);
    console.log("URL params: %s", urlParams);

    // get prolific parameters
    userID = urlParams.get("PROLIFIC_PID");
    studyID = urlParams.get("STUDY_ID");
    sessionID = urlParams.get("SESSION_ID");
    console.log("User ID: %s, study ID: %s, session ID: %s", userID, studyID, sessionID);
    if (userID === null || studyID === null || sessionID === null) {
        alert(
            "This link is not valid. Please return to Prolific and click the link again."
        );
        return;
    }

    // fetch class labels and corresponding paths to icons
    ({ labels, iconPaths } = await getLabels());
    console.log("Labels: %s", labels);
    console.log("Icon paths: %s", iconPaths);
    // shuffleArrays(labels, iconPaths); // shuffle labels and icon paths

    // fetch image paths and task ID
    ({ imgPaths, taskID, lastTrialIdx } = await getImgPaths());
    console.log("Task ID: %s", taskID);
    console.log("Image paths: %s", imgPaths);
    console.log("Last trial index: %s", lastTrialIdx);

    // set total number of trials
    numTrials = imgPaths.length;
    console.log("Num trials: %s", numTrials);

    stateIdx = -1;
    labelReviewIdx = 0;
    trialIdx = lastTrialIdx + 1;
    // reset states if no trials done so far
    if (lastTrialIdx === -1) {
        states = [
            "consent",
            "card",
            "instr1",
            "instr2",
            "label-review",
            "instr3",
            "trial",
            "done",
        ];
    } else if (lastTrialIdx < numTrials - 1) {
        states = [
            "continue",
            "card",
            "instr3",
            "trial",
            "done",
        ];
    } else if (lastTrialIdx == numTrials - 1) {
        states = [
            "done",
        ];
    } else {
        alert("Invalid value for last trial index!");
    }

    numLabelReviewScreens = Math.floor(labels.length / (numLabelReviewCols * numLabelReviewRows));
    console.log("Num label review screens: %s", numLabelReviewScreens);

    // run updateState() once the DOM has loaded
    if (document.readyState !== "loading") {
        updateState();
    } else {
        document.addEventListener("DOMContentLoaded", function() {
            updateState();
        });
    }
}

// update and display new page state
function updateState() {
    // hide old page
    const oldState = states[stateIdx];
    if (stateIdx >= 0) {
        const prevElement = document.getElementById(oldState);
        prevElement.style.display = "none";
    }

    // update trialIdx and stateIdx to new values
    if (oldState === "trial" && trialIdx < numTrials - 1) {
        // state remains 'trial'
        trialIdx++;
        console.log("New trial index is %s (0-indexed)", trialIdx);
    } else if (oldState === "label-review" && labelReviewIdx < numLabelReviewScreens - 1) {
        // state remains 'label-review'
        labelReviewIdx++;
        console.log("New label review index is %s (0-indexed)", labelReviewIdx);
    } else {
        // state changes to next state in states list
        stateIdx++;
        console.log("New state index is %s (0-indexed)", stateIdx);
    }

    // display new state
    const newState = states[stateIdx];
    if (newState == "continue") {
        updateContinueScreen();
    } else if (newState === "consent") {
        updateConsentScreen();
    } else if (newState === "card") {
        updateCardScreen();
    } else if (newState.includes("instr")) {
        updateInstrScreen(newState);
    } else if (newState === "label-review") {
        updateLabelReviewScreen();
    } else if (newState === "trial") {
        updateTrialScreen();
    } else if (newState === "done") {
        updateDoneScreen();
    };

    // show new page
    const newElement = document.getElementById(newState);
    newElement.style.display = "block";
}

// update continue screen
function updateContinueScreen() {
    // update trial progess on screen
    document.getElementById("num-completed-trials").textContent = lastTrialIdx + 1; // 1-indexed for client
    document.getElementById("num-trials-continue").textContent = numTrials;
    // button functionality
    const btn = document.getElementById("continue-btn");
    btn.addEventListener("click", () => {
        btn.disabled = true;
        console.log("Continue submit button clicked");
        updateState();
    });
    console.log("Updated continue screen");
}

// update consent screen
function updateConsentScreen() {
    // button functionality
    const btn = document.getElementById("consent-btn");
    btn.addEventListener("click", () => {
        btn.disabled = true;
        console.log("Consent submit button clicked");
        updateState();
    });
    // prevent submit until checkbox is checked
    const checkbox = document.getElementById("consent-checkbox");
    checkbox.checked = false;
    btn.disabled = true;
    checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
            btn.disabled = false;
        } else {
            btn.disabled = true;
        }
    });
    console.log("Updated consent screen");
}

// update card screen
function updateCardScreen() {
    $(() => {
        $("#card-slider").slider({ value: 50, step: 0.1 });
    });

    $(document).ready(() => {
        $("#card-slider").on("slide", (_, ui) => {
            // change size of card on screen
            const cardWidth = ui.value + "%";
            $("#card-img").css({ "width": cardWidth });
        });
    });

    // if submit button is clicked, send response to server
    const btn = document.getElementById("card-btn");
    btn.addEventListener("click", async () => {
        btn.disabled = true;
        console.log("Card button clicked");
        // get pixel width of card
        cardWidthPix = $("#card-img").width();
        console.log("Card width (pix): %s", cardWidthPix);
        updateState();
    });

    // prevent submit until checkbox is checked
    const checkbox1 = document.getElementById("fullscreen-checkbox");
    const checkbox2 = document.getElementById("card-checkbox");
    checkbox1.checked = false;
    checkbox2.checked = false;
    btn.disabled = true;
    checkbox1.addEventListener("change", () => {
        if (checkbox1.checked && checkbox2.checked) {
            btn.disabled = false;
        } else {
            btn.disabled = true;
        }
    });
    checkbox2.addEventListener("change", () => {
        if (checkbox1.checked && checkbox2.checked) {
            btn.disabled = false;
        } else {
            btn.disabled = true;
        }
    });
    console.log("Updated card screen");
}

// update instr screen
function updateInstrScreen(state) {
    // button functionality
    const btn = document.getElementById(`${state}-btn`);
    // don't allow clicks until a few seconds have passed
    btn.disabled = true;
    setTimeout(() => {
        btn.disabled = false;
    }, 0 * 1000);
    // click moves to next screen
    btn.addEventListener("click", () => {
        btn.disabled = true;
        console.log("Instructions next button clicked");
        updateState();
    });
    console.log("Updated instructions screen for %s", state);
}

// update label review screen
function updateLabelReviewScreen(forceSetup = false) {
    // setup trial screen if it is the first trial
    if (labelReviewIdx === 0 || forceSetup === true) {
        setupLabelReviewScreen();
    }

    // update labels and icons in table
    const table = document.getElementById("label-review-pane").querySelector("table");
    const numLabelsPerScreen = numLabelReviewCols * numLabelReviewRows;
    for (let r = 0; r < numLabelReviewRows; r++) {
        for (let c = 0; c < numLabelReviewCols; c++) {
            // offset
            const labelIdx = labelReviewIdx * numLabelsPerScreen + (r * numLabelReviewCols + c);
            // get a reference to the div element in the cell
            const tdDiv = table.rows[r].cells[c].querySelector('div');
            // update the img src and span textContent in the div element
            tdDiv.querySelector('img').src = iconPaths[labelIdx];
            tdDiv.querySelector('span').textContent = labels[labelIdx];
            // change background color back to base color
            const td = table.rows[r].cells[c];
            td.style.backgroundColor = baseColor;
        }
    }

    // update numbers
    document.getElementById("label-idx-lo").textContent = labelReviewIdx * numLabelsPerScreen + 1; // 1-indexed for client
    document.getElementById("label-idx-hi").textContent = (labelReviewIdx + 1) * numLabelsPerScreen; // 1-indexed for client
    console.log("Updated label review screen");

    // reset click state because new page of labels
    for (let r = 0; r < numLabelReviewRows; r++) {
        labelReviewClickedState[r].fill(false);
    }
    // disable submit button
    const btn = document.getElementById("label-review-btn");
    btn.disabled = true;
}

// set up label review screen
function setupLabelReviewScreen() {
    // turn submit button off
    const btn = document.getElementById("label-review-btn");

    // if submit button is clicked, send response to server
    btn.addEventListener("click", async () => {
        btn.disabled = true;
        console.log("Label review next button clicked");
        updateState();
    });

    // create label pane
    const table = document.createElement("table");
    table.style.width = "80%";
    table.style.height = "40vh";
    table.style.margin = "auto";
    for (let r = 0; r < numLabelReviewRows; r++) {
        const tr = document.createElement("tr");
        for (let c = 0; c < numLabelReviewCols; c++) {
            // create cell of table for one label
            const td = document.createElement("td");
            const tdHeight = 100 / numLabelReviewRows;
            const tdWidth = 100 / numLabelReviewCols;
            td.style.height = `${tdHeight}%`;
            td.style.width = `${tdWidth}%`;
            td.style.padding = "0.5vw";
            td.style.border = "1px solid";
            td.style.borderRadius = "1vw";

            // make cell look like a button
            td.style.cursor = "pointer";

            // create a new div element for vertical alignment
            var tdDiv = document.createElement("div");
            tdDiv.style.display = "flex";
            tdDiv.style.flexDirection = "column";
            tdDiv.style.alignItems = "center";

            // create a new img element and set its attributes
            const icon = document.createElement("img");
            icon.src = "";
            icon.style.width = "50%";
            icon.style.height = "50%";

            // create a new p element and set its content
            const labelSpan = document.createElement("span");
            labelSpan.textContent = "";
            labelSpan.style.textAlign = "center";
            labelSpan.style.fontSize = "1.5vw";
            labelSpan.style.paddingTop = "0.5vw";

            // append the img and p elements to the div element
            tdDiv.appendChild(icon);
            tdDiv.appendChild(labelSpan);

            // append the div element to td
            td.appendChild(tdDiv);

            // append td to tr
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }

    // make cells turn green and make next button available when all cells clicked
    labelReviewClickedState = [];
    for (let r = 0; r < numLabelReviewRows; r++) {
        labelReviewClickedState[r] = [];
        for (let c = 0; c < numLabelReviewCols; c++) {
            labelReviewClickedState[r][c] = false;
        }
    }
    for (let r = 0; r < numLabelReviewRows; r++) {
        for (let c = 0; c < numLabelReviewCols; c++) {
            const td = table.rows[r].cells[c];
            td.addEventListener("click", () => {
                labelReviewClickedState[r][c] = true;
                // change color of cell
                td.style.backgroundColor = selectColor;
                // activate submit button
                if (labelReviewClickedState.every(row => row.every(e => e === true))) {
                    btn.disabled = false;
                }
            });
        }
    }

    // finally append this table to the page
    const labelReviewPane = document.getElementById("label-review-pane");
    labelReviewPane.appendChild(table);

    console.log('Created label review screen');
}

// update trial screen with next trial image
function updateTrialScreen() {
    // setup trial screen if it is the first trial
    if (trialIdx === lastTrialIdx + 1) {
        setupTrialScreen();
    }

    // update order of labels
    // shuffleArrays(labels, iconPaths); // shuffle labels and icon paths
    const table = document.getElementById("label-tbl");
    const responseBox = document.getElementById("response-box");
    for (let r = 0; r < numLabelRows; r++) {
        for (let c = 0; c < numLabelCols; c++) {
            const td = table.rows[r].cells[c];
            const labelIdx = r * numLabelCols + c;
            // label
            const labelSpan = td.querySelector("span");
            const label = labels[labelIdx] || "";
            labelSpan.textContent = label;
            // img
            const icon = td.querySelector("img");
            icon.src = iconPaths[labelIdx] || "";

            // populate response box if button is clicked
            if (label !== "") {
                td.style.cursor = "pointer";
                td.addEventListener("click", () => {
                    responseBox.value = label;
                    btn.disabled = false;
                    responseBox.focus();
                    // color everything base color
                    const tds = table.querySelectorAll("td");
                    tds.forEach(td => {
                        td.style.backgroundColor = baseColor;
                    });
                    // color selected label
                    const labelIdx = labels.indexOf(label);
                    const r = Math.floor(labelIdx / numLabelCols);
                    const c = labelIdx % numLabelCols;
                    const td = table.rows[r].cells[c];
                    td.style.backgroundColor = selectColor;
                });
            }

        }
    }

    // make button unclickable
    // safeguards against user accidently double clicking submit button
    const btn = document.getElementById("submit-trial-btn");
    btn.disabled = true;

    // update trial progess on screen
    document.getElementById("trial-idx").textContent = trialIdx + 1; // 1-indexed for client
    document.getElementById("num-trials").textContent = numTrials;

    // update image shown to user
    document.getElementById("trial-img").src = imgPaths[trialIdx];

    // update response box
    responseBox.value = "";
    responseBox.scrollTop = 0;
    responseBox.focus();

    // color everything base color
    const tds = table.querySelectorAll("td");
    tds.forEach(td => {
        td.style.backgroundColor = baseColor;
    });
    console.log("Updated trial screen");
}

// sets up trial screen by creating table of labels
// and icons, as well as functioning response box. called on
// the first trial only
function setupTrialScreen() {
    // turn submit button off
    const btn = document.getElementById("submit-trial-btn");

    // make all images same size for different screens
    const pixPerInch = cardWidthPix / cardWidthInch;
    console.log("Card width (pix): %s, PPI: %s", cardWidthPix, pixPerInch);
    const trialImgWidthInch = pixPerInch * 5;
    const trialImg = document.getElementById("trial-img");
    trialImg.style.width = `${trialImgWidthInch}px`;

    // if submit button is clicked, send response to server
    btn.addEventListener("click", async () => {
        // if button already disabled, then that means button already clicked
        if (btn.disabled) {
            return;
        }
        // disable button to prevent double clicking
        btn.disabled = true;
        // store response to trial on server
        const imgPath = imgPaths[trialIdx];
        const selLabel = document.getElementById("response-box").value;
        // if label is not in list of labels, move on
        if (!labels.includes(selLabel)) {
            alert(`Your response "${selLabel}" is not one of the 50 options. Please try again.`);
        } else {
            // wait for server to finish storing data
            await submitTrialResponse(imgPath, selLabel, trialIdx);
            // go to next state
            updateState();
        }
    });

    // create label table
    const table = document.getElementById("label-tbl");
    for (let r = 0; r < numLabelRows; r++) {
        const tr = document.createElement("tr");
        for (let c = 0; c < numLabelCols; c++) {
            // create cell of table for one label
            const td = document.createElement("td");
            td.style.border = "1px solid";
            td.style.borderRadius = "0.5vw";
            td.style.margin = "0";
            // cell height and width
            const tdHeight = 100 / numLabelRows;
            const tdWidth = 100 / numLabelCols;
            td.style.height = `${tdHeight}%`;
            td.style.width = `${tdWidth}%`;

            // create a new div element for vertical alignment
            var tdDiv = document.createElement("div");
            tdDiv.style.display = "flex";
            tdDiv.style.flexDirection = "column";
            tdDiv.style.alignItems = "center";

            // create a new img element and set its attributes
            const icon = document.createElement("img");
            icon.src = "";
            icon.style.width = "30%";
            icon.style.height = "30%";

            // create a new p element and set its content
            const labelSpan = document.createElement("span");
            labelSpan.textContent = "";
            labelSpan.style.textAlign = "center";
            labelSpan.style.fontSize = "1vw";
            labelSpan.style.paddingTop = "0.2vw";

            // append the img and p elements to the div element
            tdDiv.appendChild(icon);
            tdDiv.appendChild(labelSpan);

            // append the div element to the td element
            td.appendChild(tdDiv);

            // append td to tr
            tr.appendChild(td);

            // hide if no label
            const labelIdx = r * numLabelCols + c;
            if (labelIdx >= iconPaths.length) {
                td.style.display = "none";
            }
        }
        table.appendChild(tr);
    }

    // response box autocomplete and other functionality
    $(function() {
        $("#response-box").autocomplete({
            source: (req, res) => {
                const matcher = new RegExp("(^|\\s)" + $.ui.autocomplete.escapeRegex(req.term), "i");
                // const matcher = new RegExp($.ui.autocomplete.escapeRegex(req.term), "i");
                res($.grep(labels, (item) => {
                    return matcher.test(item);
                }));
            },
            autoFocus: true,
            search: () => {
                const enteredText = document.getElementById("response-box").value;
                // color everything base color
                const tds = table.querySelectorAll("td");
                tds.forEach(td => {
                    td.style.backgroundColor = baseColor;
                });
                if (labels.includes(enteredText)) {
                    // color selected label
                    const labelIdx = labels.indexOf(enteredText);
                    const r = Math.floor(labelIdx / numLabelCols);
                    const c = labelIdx % numLabelCols;
                    const td = table.rows[r].cells[c];
                    td.style.backgroundColor = selectColor;
                    // allow submission
                    btn.disabled = false;
                }
                else {
                    // disable submit button
                    btn.disabled = true;
                }
            },
            select: (event, ui) => {
                // color everything base color
                const tds = table.querySelectorAll("td");
                tds.forEach(td => {
                    td.style.backgroundColor = baseColor;
                });
                // color selected label
                const label = ui.item.label;
                const labelIdx = labels.indexOf(label);
                const r = Math.floor(labelIdx / numLabelCols);
                const c = labelIdx % numLabelCols;
                const td = table.rows[r].cells[c];
                td.style.backgroundColor = selectColor;
                // disable submit button
                btn.disabled = false;
            },
            delay: 0,
        });
    });

    console.log("Created trial screen");
}

// update done screen with completion code
async function updateDoneScreen() {
    // // done code is just the MD5 hash of the taskID
    // const hash = await computeHash(taskID);
    // const doneCode = hash.substring(0, 4);

    // prolific completion code
    const doneCode = "C18RVK1H";
    document.getElementById("done-code").textContent = doneCode;
    console.log("Updated done screen");
}

// shuffle array
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// error screen
function errorScreen() {
    document.body.innerHTML = "<p>Error</p>";
}

// shuffle arrays
function shuffleArrays(arr1, arr2) {
    if (arr1.length !== arr2.length) {
        throw new Error("Arrays must be of same length");
    }

    for (let i = arr1.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr1[i], arr1[j]] = [arr1[j], arr1[i]];
        [arr2[i], arr2[j]] = [arr2[j], arr2[i]];
    }

    return [arr1, arr2];
}

//
// Fetch functions to server.
//

// fetch label names and icon paths from server
async function getLabels() {
    const response = await fetch("/get-labels", {
        method: "GET",
    });
    // check if status is in 2xx range
    if (!response.ok) {
        alert(`Error getting class labels and icon paths: ${error}`);
        errorScreen();
        return null;
    }
    // return fetched data
    const result = await response.json();
    return {
        labels: result.labels,
        iconPaths: result.iconPaths,
    };
}

// fetch img paths from server for a specific task ID
async function getImgPaths() {
    const response = await fetch("/get-img-paths", {
        method: "POST",
        body: JSON.stringify({
            userID: userID,
            studyID: studyID,
            sessionID: sessionID,
        }),
        headers: {
            "Content-Type": "application/json",
        },
    });
    // check if status is in 2xx range
    if (!response.ok) {
        alert("No tasks are available at this time. Please check back again later.");
        errorScreen();
        return null;
    }
    // return fetched paths
    const result = await response.json();
    return {
        imgPaths: result.imgPaths,
        taskID: result.taskID,
        lastTrialIdx: result.lastTrialIdx,
    };
}

// submit trial response to server
async function submitTrialResponse(imgPath, selLabel, trialIdx) {
    const response = await fetch("/submit-response", {
        method: "POST",
        body: JSON.stringify({
            userID: userID,
            taskID: taskID,
            imgPath: imgPath,
            selLabel: selLabel,
            trialIdx: trialIdx,
        }),
        headers: {
            "Content-Type": "application/json",
        },
    });

    // check if status is in the 2xx range
    if (!response.ok) {
        alert("Error submitting trial response");
        errorScreen();
        return false;
    };
    return true;
}

// hash value
async function computeHash(val) {
    const response = await fetch("/get-hash", {
        method: "POST",
        body: JSON.stringify({
            val: val,
        }),
        headers: {
            "Content-Type": "application/json",
        },
    });
    // check if status is in 2xx range
    if (!response.ok) {
        alert("Error getting hash");
        errorScreen();
        return null;
    };
    // return fetched data
    const result = await response.json();
    return result.hash;
}
