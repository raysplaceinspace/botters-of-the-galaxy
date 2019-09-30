using System;
using System.Collections.Generic;

namespace BOTG_Refree
{
    public class MovingEntity : Point
    {
        public double vx;
        public double vy;
        public double forceVX;
        public double forceVY;

        public MovingEntity(double x, double y, double vx, double vy) : base(x, y)
        {
            this.vx = vx;
            this.vy = vy;
        }

        public Point targetPointOfMovement()
        {
            return new Point(x + vx, y + vy);
        }

        public void moveIgnoreEdges(double t)
        {
            this.x += vx * t;
            this.y += vy * t;
        }

        public virtual void move(double t)
        {
            move(x + vx * t, y + vy * t);
        }

        // Move the point to x and y
        public void move(double x, double y)
        {
            this.x = x;
            this.y = y;
            if (this.x < 30) this.x = 30;
            if (this.y < 30) this.y = 30;
            if (this.x >= Const.MAPWIDTH - 30) this.x = Const.MAPWIDTH - 30;
            if (this.y >= Const.MAPHEIGHT - 30) this.y = Const.MAPHEIGHT - 30;
        }

        // Move the point to an other point for a given distance
        public void moveTo(Point p, double distance)
        {
            double d = Distance(p);

            if (d < Const.EPSILON)
            {
                return;
            }

            double dx = p.x - x;
            double dy = p.y - y;
            double coef = distance / d;

            move(x + dx * coef, y + dy * coef);
        }

        public double getCollisionTime(MovingEntity entity, double radius)
        {
            // Check instant collision
            if (this.Distance(entity) <= radius)
            {
                return 0.0;
            }

            // Fixes rounding errors.
            radius -= Const.EPSILON;

            // Both units are motionless
            if (this.vx == 0.0 && this.vy == 0.0 && entity.vx == 0.0 && entity.vy == 0.0)
            {
                return -1;
            }

            // Change referencial
            // Unit u is not at point (0, 0) with a speed vector of (0, 0)
            double x2 = this.x - entity.x;
            double y2 = this.y - entity.y;
            double r2 = radius;
            double vx2 = this.vx - entity.vx;
            double vy2 = this.vy - entity.vy;

            double a = vx2 * vx2 + vy2 * vy2;

            if (a <= 0.0)
            {
                return -1;
            }

            double b = 2.0 * (x2 * vx2 + y2 * vy2);
            double c = x2 * x2 + y2 * y2 - r2 * r2;
            double delta = b * b - 4.0 * a * c;

            if (delta < 0.0)
            {
                return -1;
            }

            double t = (-b - Math.Sqrt(delta)) / (2.0 * a);

            if (t <= 0.0)
            {
                return -1;
            }

            return t;
        }
    }
}
