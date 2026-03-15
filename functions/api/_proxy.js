function backendOrigin(env) {
  const origin = env.BACKEND_ORIGIN;
  if (!origin) {
    throw new Error('BACKEND_ORIGIN is not configured');
  }
  return origin.replace(/\/$/, '');
}

export async function proxyRequest(context) {
  const { request, env } = context;
  const incoming = new URL(request.url);
  const target = backendOrigin(env) + incoming.pathname + incoming.search;

  const headers = new Headers(request.headers);
  headers.delete('host');
  headers.delete('cf-connecting-ip');
  headers.delete('cf-ipcountry');
  headers.delete('cf-ray');
  headers.delete('x-forwarded-proto');
  headers.set('x-forwarded-host', incoming.host);
  headers.set('x-forwarded-proto', incoming.protocol.replace(':', ''));

  const init = {
    method: request.method,
    headers,
    redirect: 'manual',
    body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
  };

  const resp = await fetch(target, init);
  return new Response(resp.body, {
    status: resp.status,
    statusText: resp.statusText,
    headers: new Headers(resp.headers),
  });
}
