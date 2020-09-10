class Avg_Color:

    def get_avg_pix(self, pixels):
        red = 0
        green = 0
        blue = 0
        for x in pixels:
            red += x[0]
            green += x[1]
            blue += x[2]

        average = (red/len(pixels), green/len(pixels), blue/len(pixels))
        return average

