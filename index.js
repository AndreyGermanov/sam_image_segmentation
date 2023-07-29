document.addEventListener("DOMContentLoaded", () => {
    let file = null;
    let contour = null;
    document.querySelector("#uploadImage").addEventListener("click", () => {
        const input = document.createElement("input");
        input.setAttribute("type", "file");
        input.addEventListener("change", (event) => {
            if (!event.target.files.length) {
                return
            }
            file = event.target.files[0];
            uploadImage(file);
        })
        document.body.appendChild(input);
        input.click();
        document.body.removeChild(input);
    })

    function uploadImage(file) {
        return new Promise(resolve => {
            const img = new Image();
            img.src = window.URL.createObjectURL(file);
            img.onload = () => {
                const canvas = document.querySelector("canvas");
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img,0,0);
                resolve();
            }
        })
    }

    let is_loading = false;
    document.querySelector("#image").addEventListener("click", async(event) => {
        if (is_loading) {
            return
        }
        const canvas = document.querySelector("canvas");
        canvas.style.cursor = "wait";
        const form = new FormData();
        form.append("image",file,file.name);
        form.append("x", event.offsetX);
        form.append("y", event.offsetY);
        is_loading = true;
        const response = await fetch("/segment", {
            method: "POST",
            body: form,
        })
        const json = await response.json();
        contour = json.contour;
        is_loading = false;
        canvas.style.cursor = "pointer";
        await uploadImage(file);
        const ctx = canvas.getContext("2d");
        ctx.strokeStyle = "#00ff00"
        ctx.lineWidth = 3;
        ctx.beginPath();
        contour.forEach(([x,y]) => {
            ctx.lineTo(x,y);
        })
        ctx.closePath();
        ctx.stroke();
    })

    document.querySelector("#extractObject").addEventListener("click", () => {
        const tmpCanvas = document.createElement("canvas");
        const img = new Image();
        img.src = window.URL.createObjectURL(file)
        img.onload = () => {
            tmpCanvas.width = img.width;
            tmpCanvas.height = img.height;
            let ctx = tmpCanvas.getContext("2d");
            ctx.beginPath();
            contour.forEach(([x,y]) => {
                ctx.lineTo(x,y);
            })
            ctx.closePath();
            ctx.clip();
            ctx.drawImage(img,0,0)
            const minx = contour.map(([x,y]) => x).reduce((accum,x) => x<accum ? x : accum)
            const miny = contour.map(([x,y]) => y).reduce((accum,y) => y<accum ? y : accum)
            const maxx = contour.map(([x,y]) => x).reduce((accum,x) => x>accum ? x : accum)
            const maxy = contour.map(([x,y]) => y).reduce((accum,y) => y>accum ? y : accum)
            let destCanvas = document.getElementById("extractedImage");
            if (destCanvas) {
                document.body.removeChild(destCanvas);
            }
            destCanvas = document.createElement("canvas");
            destCanvas.width = maxx-minx;
            destCanvas.id = "extractedImage"
            destCanvas.height = maxy-miny;
            ctx = destCanvas.getContext("2d");
            ctx.drawImage(tmpCanvas, minx, miny, maxx-minx, maxy-miny,0,0,destCanvas.width,destCanvas.height);
            document.body.appendChild(destCanvas);
        }
    })
})
