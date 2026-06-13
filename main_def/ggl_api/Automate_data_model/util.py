import os
import time
from PIL import Image

def fullpage_screenshot(sb, file,link,row):
    try:
        # This is for sign-in, row != 1 is for the pic of where the sign-in itself
        if row !=0:
            if (row >= 3 and row <= 4) or (row >= 24 and row <= 25):
                sb.sleep(25)
            sb.sleep(8)
            sb.driver.execute_script("document.getElementById('social-google').click()")
            sb.sleep(8)
            # sb.open("https://www.google.com/gmail/about/")
            # sb.click('a[data-action="sign in"]')
            sb.type('input[type="email"]', "petermrdat@gmail.com")
            sb.click('button:contains("Next")')
            sb.sleep(8)
            sb.type('input[type="password"]', "Aa@123455")
            sb.click('button:contains("Next")')
            sb.sleep(25)
            sb.driver.get(link)
            sb.sleep(5)

    except Exception as e:
        print(e)
    sb.sleep(10)
    print("Starting chrome full page screenshot workaround ...")
    total_width = sb.driver.execute_script("return document.body.offsetWidth")
    total_height = sb.driver.execute_script("return document.body.parentNode.scrollHeight")
    viewport_width = sb.driver.execute_script("return document.body.clientWidth")
    viewport_width = viewport_width*2
    total_width = total_width*2
    viewport_height = sb.driver.execute_script("return window.innerHeight")
    print("Total: ({0}, {1}), Viewport: ({2},{3})".format(total_width, total_height,viewport_width,viewport_height))
    rectangles = []

    i = 0
    while i < total_height:
        ii = 0
        top_height = i + viewport_height

        if top_height > total_height:
            top_height = total_height

        while ii < total_width:
            top_width = ii + viewport_width

            if top_width > total_width:
                top_width = total_width

            print("Appending rectangle ({0},{1},{2},{3})".format(ii, i, top_width, top_height))
            rectangles.append((ii, i, top_width,top_height))

            ii = ii + viewport_width

        i = i + viewport_height

    stitched_image = Image.new('RGB', (total_width, total_height))
    previous = None
    part = 0

    for rectangle in rectangles:
        if not previous is None:
            sb.driver.execute_script("window.scrollTo({0}, {1})".format(rectangle[0], rectangle[1]))
            time.sleep(2)
            try:
                # sb.driver.execute_script("document.getElementById('header').setAttribute('style', 'position: absolute; top: 0px;');")
                sb.driver.execute_script(
                    "document.getElementsByClassName('sc-1efeg7m-0 iqfIon').setAttribute('style', 'position: absolute; top: 0px;');")


                time.sleep(2)
                print("Scrolled To ({0},{1})".format(rectangle[0], rectangle[1]))
                time.sleep(2)
            except:
                print("Scrolled To ({0},{1})".format(rectangle[0], rectangle[1]))
                time.sleep(2)
                pass

        file_name = "part_{0}.png".format(part)


        print("Capturing {0} ...".format(file_name))


        sb.driver.get_screenshot_as_file(file_name)
        screenshot = Image.open(file_name)

        if rectangle[1] + viewport_height > total_height:
            offset = (rectangle[0], total_height - viewport_height)
        else:
            offset = (rectangle[0], rectangle[1])

        print("Adding to stitched image with offset ({0}, {1})".format(offset[0],offset[1]))
        stitched_image.paste(screenshot, offset)

        del screenshot
        os.remove(file_name)
        part = part + 1
        previous = rectangle

    stitched_image.save(file)
    print("Finishing chrome full page screenshot workaround...")
    return True