import OpenGL.GL as gl
import moderngl

ctx = moderngl.create_standalone_context()
tex3d = ctx.texture3d((10, 10, 10), 4, dtype='f2')

dummy_tex = ctx.texture((10, 10), 4, dtype='f2')
fbo = ctx.framebuffer(color_attachments=[dummy_tex])

gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo.glo)
gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, tex3d.glo, 0)

status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
print("FBO Status:", status == gl.GL_FRAMEBUFFER_COMPLETE)
print("Wrapped FBO:", fbo)
