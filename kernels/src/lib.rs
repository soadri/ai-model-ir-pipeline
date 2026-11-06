#![allow(unexpected_cfgs)]
#![expect(clippy::type_complexity)] // Necessary for mapping expressions.
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod kernel;

/// 0.6 migration helper: convert typed slice to Vec<u8>
pub fn to_bytes<T>(data: &[T]) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    }.to_vec()
}
