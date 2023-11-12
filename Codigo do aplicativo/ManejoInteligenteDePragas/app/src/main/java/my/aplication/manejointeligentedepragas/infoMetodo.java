package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;

//import android.support.v4.content.ContextCompat;
import androidx.core.content.ContextCompat;
//import android.support.v4.view.ViewPager;
import androidx.viewpager.widget.ViewPager;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.ImageAdapter.ViewPagerAdapter;

import com.example.manejointeligentedepragas.R;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class infoMetodo extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener
{

    int codMetodo;

    TextView tvNome;
    TextView tvMateriaisNecessarios;
    TextView tvModoDePreparo;
    TextView tvIntervaloAplicacao;
    TextView tvEfeitoColateral;
    TextView tvObservacoes;
    TextView tvAtuacao;
    LinearLayout sliderDotspanel;

    //pontos indicadores
    private int dotscount;
    private ImageView[] dots;

    String Nome;
    String MateriaisNecessarios;
    String ModoDePreparo;
    int IntervaloAplicacao ;
    String EfeitoColateral;
    String Observacoes;
    String Atuacao;

    ArrayList<String> urlsMetodos = new ArrayList<>();

    private DrawerLayout drawerLayout;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info_metodo);

        tvNome = findViewById(R.id.InfoMetodoNome);
        tvMateriaisNecessarios = findViewById(R.id.InfoMetodoMateriais);
        tvModoDePreparo = findViewById(R.id.InfoMetodoPreparo);
        tvIntervaloAplicacao = findViewById(R.id.InfoMetodoIntervalo);
        tvEfeitoColateral = findViewById(R.id.IInfoMetodoEfeito);
        tvObservacoes = findViewById(R.id.InfoMetodoObs);
        tvAtuacao = findViewById(R.id.InfoMetodoAtuacao);
        sliderDotspanel = findViewById(R.id.SliderDotsMetodo);

        codMetodo = getIntent().getIntExtra("Cod_Metodo",0);


        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_infoMetodo);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_infoMetodo);
        NavigationView navigationView = findViewById(R.id.nav_view_infoMetodo);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);


        ViewPager viewPager = findViewById(R.id.ViewPagerMetodo);
        ResgatarUrlMetodos(viewPager, codMetodo);
        ResgataMetodos(codMetodo);

        viewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
            @Override
            public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {

            }

            @Override
            public void onPageSelected(int position) {

                for(int i = 0; i< dotscount; i++){
                    dots[i].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.non_active_dot));
                }

                dots[position].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.active_dot));

            }

            @Override
            public void onPageScrollStateChanged(int state) {

            }
        });
    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Intent i= new Intent(this, Perfil.class);
                startActivity(i);
                break;
            case R.id.drawerProp:
                Intent prop= new Intent(this, Propriedades.class);
                startActivity(prop);
                break;

            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;

            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }


    public void ResgataMetodos(int codM){
        Utils u = new Utils();
        if(!u.isConected(getBaseContext()))
        {
            Toast.makeText(this,"Habilite a conexão com a internet!", Toast.LENGTH_LONG).show();
        }else { // se tem acesso à internet
            String url = "https://mip.software/phpapp/infoMetodo.php?Cod_Metodo="+codM;

            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i< array.length(); i++){
                            JSONObject obj = array.getJSONObject(i);

                            Nome = obj.getString("Nome");
                            MateriaisNecessarios = obj.getString("MateriaisNecessarios");
                            ModoDePreparo = obj.getString("ModoDePreparo");
                            IntervaloAplicacao  = obj.getInt("IntervaloAplicacao");
                            EfeitoColateral = obj.getString("EfeitoColateral");
                            Observacoes = obj.getString("Observacoes");
                            Atuacao = obj.getString("Atuacao");
                        }

                         tvNome.setText("Nome: "+Nome);
                         tvMateriaisNecessarios.setText("Materiais necesários: "+MateriaisNecessarios);
                         tvModoDePreparo.setText("Modo de preparo: "+ModoDePreparo);
                         if (IntervaloAplicacao == 0){
                             tvIntervaloAplicacao.setText("Intervalo de aplicação(em dias): ver recomendação do distribuidor.");
                         }else{
                             tvIntervaloAplicacao.setText("Intervalo de aplicação(em dias): "+IntervaloAplicacao);
                         }
                         tvEfeitoColateral.setText("Efeitos colaterais: "+EfeitoColateral);
                         tvObservacoes.setText("Observações: "+Observacoes);
                         tvAtuacao.setText("Atuação: "+Atuacao);

                        setTitle("MIP² | "+Nome);
                    } catch (JSONException e) {
                        Toast.makeText(infoMetodo.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(infoMetodo.this,error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }
    }

    public void ResgatarUrlMetodos(final ViewPager viewPager, int codM){
        Utils u = new Utils();
        if (!u.isConected(getBaseContext())) {
            Toast.makeText(this, "Habilite a conexão com a internet", Toast.LENGTH_LONG).show();
        } else { // se tem acesso à internet

            String url = "https://mip.software/phpapp/resgatarFotoMetodos.php?Cod_Metodo="+codM;

            RequestQueue queue = Volley.newRequestQueue(infoMetodo.this);
            queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

                @Override
                public void onResponse(String response) {
                    //Parsing json
                    //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                    try {
                        JSONArray array = new JSONArray(response);
                        for (int i = 0; i < array.length(); i++) {
                            JSONObject obj = array.getJSONObject(i);
                            urlsMetodos.add("https://mip.software/metodos/"+obj.getString("FotoMetodo"));
                        }
                        ViewPagerAdapter adapterMetodos = new ViewPagerAdapter(infoMetodo.this,urlsMetodos);
                        viewPager.setAdapter(adapterMetodos);

                        dotscount = adapterMetodos.getCount();
                        dots = new ImageView[dotscount];


                        for(int i = 0; i < dotscount; i++){

                            dots[i] = new ImageView(infoMetodo.this);
                            dots[i].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.non_active_dot));

                            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);

                            params.setMargins(8, 0, 8, 0);

                            sliderDotspanel.addView(dots[i], params);

                        }

                        dots[0].setImageDrawable(ContextCompat.getDrawable(getApplicationContext(), R.drawable.active_dot));
                        if(urlsMetodos.size()== 0){
                            viewPager.setVisibility(View.GONE);
                            sliderDotspanel.setVisibility(View.GONE);
                        }

                    } catch (JSONException e) {
                        Toast.makeText(infoMetodo.this, e.toString(), Toast.LENGTH_LONG).show();
                    }
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Toast.makeText(infoMetodo.this, error.toString(), Toast.LENGTH_LONG).show();
                }
            }));

        }

    }

}
