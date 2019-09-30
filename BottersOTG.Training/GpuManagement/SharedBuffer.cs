namespace Telogis.RouteCloud.GPUManagement {
	public struct SharedBuffer {
		public static SharedBuffer Floats(int length) {
			return new SharedBuffer(sizeof(float) * length);
		}

		public static SharedBuffer Bytes(int length) {
			return new SharedBuffer(length);
		}

		public static SharedBuffer Ints(int length) {
			return new SharedBuffer(sizeof(int) * length);
		}

		public SharedBuffer(int size) : this() {
			Size = size;
		}

		public int Size { get; private set; }
	}
}
